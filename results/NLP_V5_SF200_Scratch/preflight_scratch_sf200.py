"""Read-only preflight of Scratch V5: syntax, generator and 64-batch backward."""
from pathlib import Path
import ast,hashlib,importlib.util,json,sys,random
import torch
from courier.common import load_dataset
from courier.nlp.neural import NeuralParserNet,HEADS,encode_texts,spec_labels
from courier.nlp.synth import spec_from_mission
p=Path(r"D:\phenikaa\results\NLP_V5_SF200_Scratch")
src=p/"train_scratch_sf200_v2.py"
ast.parse(src.read_text(encoding="utf-8"))
s=src.read_text(encoding="utf-8")
assert 'load_dataset(ROOT/"Phenikaa_Campus_Courier_2026_v3/delivery_public","train")' in s
assert '"validation")' not in s and '"test")' not in s
assert 'init_checkpoint' not in s
conf=json.loads((p/"scratch_config.json").read_text(encoding="utf-8"))
assert hashlib.sha256((p/"synth_weighted_v4.py").read_bytes()).hexdigest()==conf["generator_sha256"]
mod=importlib.util.spec_from_file_location("sf200_scratch_preflight_generator",p/"synth_weighted_v4.py")
gen=importlib.util.module_from_spec(mod);sys.modules[mod.name]=gen;mod.loader.exec_module(gen)
g=gen.Generator(seed=2026110501,holdout=False,mode="weighted")
rows=[]
for name,percentage in gen.WEIGHTED_RECIPES.items():
 g._forced_recipe=name
 for _ in range(12):
  z=g.example()
  assert len(spec_labels(z.goal,z.via,z.urgent,z.fragile))==8
  rows.append(z)
print("GENERATOR_PREFLIGHT_OK",len(rows),flush=True)
if not torch.cuda.is_available():raise SystemExit("No CUDA")
net=NeuralParserNet(dim=96,hidden=192).cuda()
net.train()
sample=[rows[i%len(rows)] for i in range(64)]
x=encode_texts([r.text for r in sample]).cuda()
lab=[spec_labels(r.goal,r.via,r.urgent,r.fragile) for r in sample]
out=net(x)
loss=sum(torch.nn.functional.cross_entropy(out[h],torch.tensor([r[h] for r in lab],device="cuda")) for h in HEADS)
loss.backward()
torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
print("SCRATCH_PREFLIGHT_OK",json.dumps({"loss":float(loss.item()),"batch":64,"gpu_peak_reserved_mib":int(torch.cuda.max_memory_reserved()/1048576)}),flush=True)
