"""V5-SF200 fine-tune from frozen V2 seed-0 checkpoint (source files untouched).

Two 200K synthetic corpora (different seeds) + 2000 real x5 per epoch.
E1 / E2 model artifacts are different new files. Checkpoint selection uses
ONLY development sets. Final holdouts are constructed/evaluated AFTER selection.
"""
from __future__ import annotations

import copy,gc,hashlib,importlib.util,json,math,os,random,shutil,sys,time
from pathlib import Path
import torch
from torch import nn
from courier.common import load_dataset
from courier.nlp import MissionParser
from courier.nlp.synth import augment_real
from courier.nlp.neural import NeuralParserNet,NeuralMissionParser,HEADS,encode_texts

ROOT=Path(r"D:\phenikaa")
RUN=ROOT/"results/NLP_V5_SemanticFusion_200K"
SOURCE=ROOT/"results/NLP_V4_R2/unified_weighted_200k_20261010_a"
CONFIG=json.loads((RUN/"sf200_config.json").read_text(encoding="utf-8"))
CHECKPOINT=Path(CONFIG["init_checkpoint"])
EPOCH_FILES=[Path(CONFIG["epoch1_data"]),Path(CONFIG["epoch2_data"])]
OUTPUT=[RUN/"sf200_e1.pt",RUN/"sf200_e2.pt",RUN/"sf200_best.pt",
        RUN/"sf200_evaluation.json"]
for path in OUTPUT:
    if path.exists():raise FileExistsError("Refuse overwrite: "+str(path))
for path in [CHECKPOINT,*EPOCH_FILES]:
    if not path.is_file():raise FileNotFoundError(str(path))
# Verify E1 bytes against independent audit. E2 is separately re-audited.
known_epoch1_sha256="da03ac6bb3498b177ed238855489212c23df895beb4b675b7920f5ea15b1645a"
if hashlib.sha256(EPOCH_FILES[0].read_bytes()).hexdigest()!=known_epoch1_sha256:
    raise RuntimeError("Unexpected epoch1 corpus mutation")

sp=importlib.util.spec_from_file_location("v5_original_trainer_helpers",ROOT/"scripts/nlp/train_neural_parser.py")
legacy=importlib.util.module_from_spec(sp);sys.modules[sp.name]=legacy;sp.loader.exec_module(legacy)
sg=importlib.util.spec_from_file_location("v5_weighted_generator_readonly",SOURCE/"synth_weighted_v4.py")
weighted=importlib.util.module_from_spec(sg);sys.modules[sg.name]=weighted;sg.loader.exec_module(weighted)

torch.set_num_threads(4)
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable. Refusing accidental long CPU training.")
DEVICE=torch.device("cuda")
seed=int(CONFIG["training_seed"])
random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
net=NeuralParserNet(dim=CONFIG["dim"],hidden=CONFIG["hidden"])
artifact=torch.load(CHECKPOINT,map_location="cpu",weights_only=False)
if artifact["config"]!=net.config:raise RuntimeError(f"Architecture mismatch {artifact['config']} vs {net.config}")
state=(artifact.get("state_dicts") or [artifact["state_dict"]])[0]
net.load_state_dict(state)
net=net.to(DEVICE)
print("TRAIN_START","V5-SF200","GPU",torch.cuda.get_device_name(0),
      "parameters",sum(p.numel() for p in net.parameters()),flush=True)
print("SOURCE_V2_SEED0",str(CHECKPOINT),"MODEL_CONFIG",net.config,flush=True)

data_root=ROOT/"Phenikaa_Campus_Courier_2026_v3/delivery_public"
real=legacy.labelled_real(load_dataset(data_root,"train"))
val_real=legacy.labelled_real(load_dataset(data_root,"validation"))
assert len(real)>0 and len(val_real)>0
# Development sets may be used to choose E1 or E2. FINAL HOLDOUT is not
# even generated until after choosing a checkpoint.
dev_v2=legacy.labelled_synthetic(2000,seed=2026101091,holdout=True)
dev_weighted=[(e.text,e.goal,e.via,e.urgent,e.fragile) for e in
              weighted.generate(1600,seed=2026101092,holdout=True,mode="weighted")]
dev_hard=[(e.text,e.goal,e.via,e.urgent,e.fragile) for e in
          weighted.generate(700,seed=2026101093,holdout=True,mode="hard")]
dev_sets={"real_validation":val_real,"v2_heldout_phrasing":dev_v2,
          "weighted_heldout":dev_weighted,"hard_heldout":dev_hard}
print("DEV_SIZES",json.dumps({k:len(v) for k,v in dev_sets.items()}),
      "REAL_TRAIN",len(real),"REAL_REPEAT",CONFIG["real_repeat"],flush=True)

def neural_score(net,rows):
    net.eval()
    model=NeuralMissionParser(net)
    texts=[row[0] for row in rows]
    preds=legacy.predict_all(model,texts,batch=128)
    result=legacy.spec_score(rows,[p.parsed for p in preds])
    # An explicit via=false-positive rate on via=None examples.
    absent=[i for i,row in enumerate(rows) if row[2] is None]
    result["via_absent_n"]=len(absent)
    result["via_false_positive_rate"]=(
       sum(preds[i].parsed.via is not None for i in absent)/len(absent)
       if absent else None)
    return result

def evaluate_dev(net):
    return {name:neural_score(net,rows) for name,rows in dev_sets.items()}

def score_from_dev(dev):
    return (.35*dev["v2_heldout_phrasing"]["all"]+
            .35*dev["weighted_heldout"]["all"]+
            .20*dev["hard_heldout"]["all"]+
            .10*dev["real_validation"]["all"])

print("BASELINE_EVALUATING",flush=True)
baseline=evaluate_dev(net)
print("BASELINE_DEV",json.dumps(baseline,ensure_ascii=False),flush=True)

lr=CONFIG["learning_rate_peak"]
optimiser=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=CONFIG["weight_decay"])
samples_per_epoch=[200000+len(real)*CONFIG["real_repeat"]]*2
assert samples_per_epoch==[210000,210000],samples_per_epoch
steps_per_epoch=math.ceil(samples_per_epoch[0]/CONFIG["batch"])
total_steps=2*steps_per_epoch
scheduler=torch.optim.lr_scheduler.OneCycleLR(
    optimiser,max_lr=lr,total_steps=total_steps,pct_start=CONFIG["pct_start"],
    anneal_strategy="cos",div_factor=25.0,final_div_factor=10000.0)
loss_fn=nn.CrossEntropyLoss()
results={"experiment":"V5-SF200","status":"COMPLETE",
   "configuration":CONFIG,"baseline_dev":baseline,"epochs":[],
   "selection_policy":"Dev only; final holdout evaluated after selection",
   "epoch_data_sha256":[hashlib.sha256(p.read_bytes()).hexdigest() for p in EPOCH_FILES],
   "trainer_source":"new results-only train_sf200.py",
   "steps_per_epoch":steps_per_epoch}
best_index=None;best_score=-1
for epoch,path in enumerate(EPOCH_FILES,1):
    print("EPOCH_DATA_LOADING",epoch,path.name,flush=True)
    synthetic=legacy.labelled_prebuilt(path)
    assert len(synthetic)==200000
    rows=synthetic + real*CONFIG["real_repeat"]
    del synthetic
    random.Random(seed+epoch).shuffle(rows)
    net.train()
    seen=0;loss_sum=0;start=time.time()
    for offset in range(0,len(rows),CONFIG["batch"]):
        chunk=rows[offset:offset+CONFIG["batch"]]
        x=encode_texts([r[0] for r in chunk]).to(DEVICE)
        y={name:t.to(DEVICE) for name,t in legacy.tensors(chunk).items()}
        logits=net(x)
        loss=sum(loss_fn(logits[name],y[name]) for name in HEADS)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(),CONFIG["gradient_clip"])
        optimiser.step();scheduler.step()
        seen+=len(chunk);loss_sum+=loss.item()*len(chunk)
        batch_index=offset//CONFIG["batch"]+1
        if batch_index==1 or batch_index%120==0 or seen==len(rows):
            print("TRAIN_PROGRESS",f"epoch={epoch}/2",
                f"batch={batch_index}/{steps_per_epoch}",
                f"rows={seen}/{len(rows)}",
                f"mean_loss={loss_sum/seen:.5f}",
                f"lr={optimiser.param_groups[0]['lr']:.8f}",
                f"elapsed_s={time.time()-start:.0f}",flush=True)
    assert seen==210000
    del rows
    gc.collect()
    dev=evaluate_dev(net)
    weighted_score=score_from_dev(dev)
    base_v2=baseline["v2_heldout_phrasing"]
    base_real=baseline["real_validation"]
    guard={
      "dev_v2_via_no_catastrophic_drop":
         dev["v2_heldout_phrasing"]["via"]>=base_v2["via"]-0.04,
      "dev_v2_all_no_catastrophic_drop":
         dev["v2_heldout_phrasing"]["all"]>=base_v2["all"]-0.035,
      "real_via_no_catastrophic_drop":
         dev["real_validation"]["via"]>=base_real["via"]-0.05,
    }
    info={"epoch":epoch,"training_loss":loss_sum/seen,"elapsed_seconds":round(time.time()-start,1),
          "score_for_selection":weighted_score,"guards":guard,"guard_pass":all(guard.values()),
          "dev_metrics":dev,"checkpoint":str(OUTPUT[epoch-1])}
    results["epochs"].append(info)
    # Each unique checkpoint is saved even when it fails a regression guard.
    assert not OUTPUT[epoch-1].exists()
    NeuralMissionParser(net).save(OUTPUT[epoch-1],
      checkpoint=True,seed=seed,epoch=epoch,training={"experiment":"V5-SF200",**CONFIG},
      metrics=info)
    print("EPOCH_RESULT",json.dumps(info,ensure_ascii=False),flush=True)
    if all(guard.values()) and weighted_score>best_score:
        best_score=weighted_score;best_index=epoch-1
    print("CHECKPOINT_SAVED",OUTPUT[epoch-1],flush=True)

# If no epoch passes the baseline regression guard, preserve the most promising
# one but explicitly REJECT promotion over V2 in report.
passed=best_index is not None
if not passed:
    best_index=max(range(len(results["epochs"])),
                   key=lambda i:results["epochs"][i]["score_for_selection"])
    print("WARNING_NO_CHECKPOINT_PASSED_REGRESSION_GUARD",flush=True)
best_epoch=best_index+1
with OUTPUT[best_index].open("rb") as fsrc,OUTPUT[2].open("xb") as fdst:
    shutil.copyfileobj(fsrc,fdst)
results["selected_epoch"]=best_epoch
results["passes_v2_regression_guards"]=passed
results["selected_checkpoint"]=str(OUTPUT[2])
print("CHECKPOINT_SELECTED",best_epoch,"guards_ok",passed,"best",OUTPUT[2],flush=True)

# Evaluate on FINAL holdouts *only after* choosing the checkpoint. They are not
# used to revise checkpoint/learning-rate/hyperparameters.
final_sets={
 "final_v2_holdout":legacy.labelled_synthetic(3000,seed=2026101171,holdout=True),
 "final_weighted_holdout":[(e.text,e.goal,e.via,e.urgent,e.fragile) for e in
      weighted.generate(2000,seed=2026101172,holdout=True,mode="weighted")],
 "final_hard_holdout":[(e.text,e.goal,e.via,e.urgent,e.fragile) for e in
      weighted.generate(1000,seed=2026101173,holdout=True,mode="hard")],
}
print("FINAL_EVALUATION",json.dumps({k:len(v) for k,v in final_sets.items()}),flush=True)
net=net.to("cpu")
del net
torch.cuda.empty_cache()
def eval_artifact(path,sets):
    raw=torch.load(path,map_location="cpu",weights_only=False)
    model=NeuralParserNet(**raw["config"])
    model.load_state_dict((raw.get("state_dicts") or [raw["state_dict"]])[0])
    model=model.to(DEVICE)
    scores={name:neural_score(model,rows) for name,rows in sets.items()}
    del model
    torch.cuda.empty_cache()
    return scores
results["final_v2_seed0"]=eval_artifact(CHECKPOINT,final_sets)
results["final_v5_sf200"]=eval_artifact(OUTPUT[2],final_sets)
results["final_caution"]="Final holdouts come from heldout phrase banks but can share structural template families with dev. Do not treat these as hidden Kaggle performance."
# JSON only created once after training completes, no old file overwritten.
with OUTPUT[3].open("x",encoding="utf-8") as f:json.dump(results,f,ensure_ascii=False,indent=2)
print("TRAIN_COMPLETED",json.dumps({"selected_epoch":best_epoch,"guards_ok":passed,
      "final_V2":results["final_v2_seed0"],"final_V5":results["final_v5_sf200"]},
      ensure_ascii=False),flush=True)
print("REPORT_SAVED",OUTPUT[3],flush=True)
