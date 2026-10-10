"""Reproduce V2 versus V5 on the same 300 official validation scenes.

Compute CV ONCE, use identical predictions, same solver and NLP threshold.
Also record V5's previously reported threshold .80. All writes create-only in
research workspace. This is evaluation, not training, no original source edits.
"""
from __future__ import annotations
import dataclasses, datetime, hashlib, json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from courier.common import load_dataset
from courier.cv import CVInput, CVPipeline, SklearnWeatherClassifier, load_annotations, load_rgb
from courier.cv.neural import SharedDetector, NeuralLegendReader, NeuralGridDetector, NeuralEdgeClassifier, NeuralNodeClassifier
from courier.cv.types import edge_key
from courier.nlp import resolve
from courier.nlp.synth import spec_from_mission
from courier.nlp.neural import HybridMissionParser, HybridThresholds
from courier.solver import load_strategy
ROOT=Path(r"D:\phenikaa")
WORK=ROOT/"results/NLP_Data_Research_20261010_a"
REPORT=WORK/"checkpoints/validation_v2_v5_same_solver_recheck_20261010.json"
DETAIL=WORK/"checkpoints/validation_v2_v5_same_solver_detail_20261010.jsonl"
for path in (REPORT,DETAIL):
    if path.exists():raise FileExistsError(str(path))
torch.set_num_threads(3)
start=time.monotonic()
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def msg(*args):
    print(*args,flush=True)
dataset_dir=ROOT/"Phenikaa_Campus_Courier_2026_v3/delivery_public"
data=load_dataset(dataset_dir,"validation")
annotations=load_annotations(dataset_dir,"validation")
truths=data.scenes
labels=np.asarray(data.labels,dtype=np.int8).reshape(-1,10)
assert len(annotations)==len(truths)==len(labels)==300
V2=ROOT/"artifacts/nlp/neural_parser_v2.pt"
V5=ROOT/"artifacts/nlp/neural_parser_v5_sf200_scratch.pt"
STRATEGIES={
  "solver_train_only":ROOT/"artifacts/solver/candidate_strategy.joblib",
  "solver_trainval":ROOT/"artifacts/solver/candidate_strategy_trainval.joblib"}
assert all(p.exists() for p in (V2,V5,*STRATEGIES.values()))
shared=SharedDetector(ROOT/"artifacts/cv/detector.pt",threshold=.4,semantic_threshold=.15,device="auto")
cv=CVPipeline(legend=NeuralLegendReader(shared),
   weather=SklearnWeatherClassifier.load(ROOT/"artifacts/cv/weather_classifier.joblib"),
   grid=NeuralGridDetector(shared),
   edges=NeuralEdgeClassifier(ROOT/"artifacts/cv/edge_net.pt"),
   nodes=NeuralNodeClassifier(ROOT/"artifacts/cv/node_net.pt"))
graphs=[]
cv_equal=[]
for idx,(ann,truth) in enumerate(zip(annotations,truths)):
    graph=cv.extract(CVInput(ann.scene_id,load_rgb(ann.image_path)))
    graphs.append(graph)
    cv_equal.append(
       (graph.grid.rows,graph.grid.cols)==(ann.graph.grid.rows,ann.graph.grid.cols)
       and graph.grid.nodes==ann.graph.grid.nodes
       and {edge_key(e):e for e in graph.edges}=={edge_key(e):e for e in ann.graph.edges}
       and set(graph.landmarks)==set(ann.graph.landmarks)
       and graph.robot_rc==ann.graph.robot_rc
       and graph.robot_heading==ann.graph.robot_heading
       and graph.weather==ann.graph.weather)
    if (idx+1)%50==0:msg("CV_SCENES",idx+1,"elapsed_s",round(time.monotonic()-start,1))
msg("CV_DONE",len(graphs),sum(cv_equal),"seconds",round(time.monotonic()-start,1))
def metric(parsed,expected):
    goal,via=spec_from_mission(expected)
    correct={"goal":parsed.goal==goal,"via":parsed.via==via,
             "urgent":parsed.urgent==expected.urgent,
             "fragile":parsed.fragile==expected.fragile}
    correct["all"]=all(correct.values())
    return correct
variant_models={
   "V2_goal099":(V2,.99),
   "V5_goal099":(V5,.99),
   "V5_goal080":(V5,.8)}
scenes_by_variant={}
nlp_metrics={}
nlp_detail={}
for name,(path,goal_threshold) in variant_models.items():
    hp=HybridMissionParser.load(path,thresholds=HybridThresholds(
        goal=goal_threshold,via=2.0,flags=2.0))
    sc={"nlp_only":[],"full":[]}
    vals={"hybrid_raw":Counter(),"neural_raw":Counter()}
    drows=[]
    for idx,(truth,graph) in enumerate(zip(truths,graphs)):
        oracle=resolve(hp.parse_for_map(truth.mission.text,truth.landmarks),truth.landmarks)
        cv_mission=resolve(hp.parse_for_map(truth.mission.text,graph.landmarks),graph.landmarks)
        sc["nlp_only"].append(dataclasses.replace(truth,mission=oracle))
        sc["full"].append(graph.to_scene(truth.scene_id,cv_mission))
        # Raw-field accuracy is non-map grounded; capture as a separate diagnostic.
        if name!="V5_goal080":
            rawrule=hp.parse(truth.mission.text)
            rawneural=hp.neural.parse(truth.mission.text)
            r=metric(rawrule,truth.mission)
            n=metric(rawneural,truth.mission)
            for key in r: vals["hybrid_raw"][key]+=r[key]
            for key in n: vals["neural_raw"][key]+=n[key]
            drows.append({"scene_index":idx,"hybrid_fields":r,"neural_fields":n})
        if (idx+1)%100==0:msg("NLP_PROGRESS",name,idx+1)
    scenes_by_variant[name]=sc
    nlp_detail[name]=drows
    nlp_metrics[name]={kind:{key:round(cnt[key]/300,6) for key in ("goal","via","urgent","fragile","all")}
       for kind,cnt in vals.items() if sum(cnt.values())>0}
    msg("NLP_DONE",name,nlp_metrics[name],"elapsed",round(time.monotonic()-start,1))
oracle_scene=tuple(truths)
cv_only_scene=tuple(graph.to_scene(truth.scene_id,truth.mission)
   for truth,graph in zip(truths,graphs))
full_results={}
predictions={}
for solver_name,solver_path in STRATEGIES.items():
    strategy=load_strategy(solver_path)
    groups={"oracle":oracle_scene,"cv_only":cv_only_scene}
    for name,sc in scenes_by_variant.items():
        groups[name+"_nlp_only"]=tuple(sc["nlp_only"])
        groups[name+"_full"]=tuple(sc["full"])
    per={}
    for scenario,scenes in groups.items():
        p=np.asarray(strategy.predict_scenes(scenes),dtype=np.int8)
        assert p.shape==(300,10),(scenario,p.shape)
        matches=p==labels
        predictions[(solver_name,scenario)]=p
        per[scenario]={"correct":int(matches.sum()),"total":int(matches.size),
           "macro_accuracy":round(float(matches.mean()),6),
           "per_robot":np.round(matches.mean(axis=0),6).tolist(),
           "scene_all_10_correct":int(matches.all(axis=1).sum()),
           "scene_accuracy_mean":round(float(matches.mean(axis=1).mean()),6)}
        msg("SCORE",solver_name,scenario,per[scenario]["correct"],"/",per[scenario]["total"],
            per[scenario]["macro_accuracy"])
    full_results[solver_name]=per
comparisons={}
for solver_name in STRATEGIES:
    for scene_kind in ("nlp_only","full"):
        a=predictions[(solver_name,"V2_goal099_"+scene_kind)]
        b=predictions[(solver_name,"V5_goal099_"+scene_kind)]
        diff=(b==labels).astype(int)-(a==labels).astype(int)
        comparisons[solver_name+"_"+scene_kind]={
          "V5minusV2_correct_predictions":int(diff.sum()),
          "V5minusV2_percentage_points":round(float(diff.mean()*100),4),
          "V5_only_correct":int(np.logical_and(b==labels,a!=labels).sum()),
          "V2_only_correct":int(np.logical_and(a==labels,b!=labels).sum()),
          "scene_differences_nonzero":int(np.count_nonzero(diff.sum(axis=1)))}
        msg("FAIR_DIFFERENCE",solver_name,scene_kind,comparisons[solver_name+"_"+scene_kind])
with DETAIL.open("x",encoding="utf-8") as f:
    for idx in range(300):
        record={"index":idx,"id":annotations[idx].scene_id,"cv_exact":bool(cv_equal[idx]),
                "systems":{},"nlp_raw":{}}
        for solver_name in STRATEGIES:
            record["systems"][solver_name]={}
            for nm in ("V2_goal099","V5_goal099","V5_goal080"):
                sc=nm+"_full"
                record["systems"][solver_name][nm]={
                    "correct_robot_count":int(np.equal(predictions[(solver_name,sc)][idx],labels[idx]).sum()),
                    "prediction":predictions[(solver_name,sc)][idx].tolist()}
        for name,rows in nlp_detail.items():
            if rows:record["nlp_raw"][name]=rows[idx]
        f.write(json.dumps(record,ensure_ascii=False)+"\n")
summary={"name":"reproducible_v2_v5_valid300_same_cv_same_solver",
 "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "valid_scenes":300,"observations":3000,"device":"auto CV and CPU NLP",
 "validated_source_files":{str(p):digest(p) for p in (V2,V5,*STRATEGIES.values())},
 "config":{"cv_detector_threshold":.4,"cv_semantic_threshold":.15,"map_aware":True,
   "repair_unreachable":True,"shared_cv":True,
   "same_threshold_comparison":{"goal":.99,"via":2.0,"flags":2.0},
   "v5_previous_threshold":{"goal":.8,"via":2.0,"flags":2.0}},
 "cv_scene_exact":{"n":sum(cv_equal),"fraction":sum(cv_equal)/300},
 "nlp_raw":nlp_metrics,"pipeline_accuracy":full_results,
 "fair_differences":comparisons,"detail_file":str(DETAIL),
 "elapsed_seconds":round(time.monotonic()-start,1),
 "limitations":["Official validation has been reused for past comparisons, NOT blind test",
    "solver_trainval may have seen validation; solver_train_only is the primary comparison",
    "NLP model might have used dev split for historical tuning; training source not audited",
    "No test or train file modified; all new artifacts only within research workspace"]}
with REPORT.open("x",encoding="utf-8") as f:json.dump(summary,f,ensure_ascii=False,indent=2)
msg("FINISHED_RECHECK",str(REPORT),"seconds",summary["elapsed_seconds"])
