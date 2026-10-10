"""Create corrected scratch trainer v2 under results; never edit v1."""
import ast
from pathlib import Path
folder=Path(r"D:\phenikaa\results\NLP_V5_SF200_Scratch")
source=(folder/"train_scratch_sf200.py").read_text(encoding="utf-8")
i=source.index('        with ckpt.open("xb") as f:')
j=source.index('        summary={"seed_index":',i)
replacement='''        with ckpt.open("xb") as f:
            torch.save({
                "kind":"courier.nlp.neural_parser","config":net.config,
                "state_dicts":[{k:v.detach().cpu() for k,v in net.state_dict().items()}],
                "checkpoint":True,"seed":model_seed,"epoch":epoch,
                "experiment":CFG["name"],
                "training":{"from_scratch":True,"official_validation_used":False},
                "synthetic_seed":manifest["data_seed"],"model_config":net.config,
            },f)
'''
source=source[:i]+replacement+source[j:]
i=source.index('    with ENSEMBLE.open("xb") as f:')
j=source.index('    report={"status":"DONE"',i)
replacement='''    with ENSEMBLE.open("xb") as f:
        torch.save({
            "kind":"courier.nlp.neural_parser",
            "config":{"dim":96,"hidden":192},
            "state_dicts":[{k:v.detach().cpu() for k,v in net.state_dict().items()} for net in nets],
            "experiment":CFG["name"],"model_seed":MODELS,
            "checkpoint_selection":"epoch15 fixed, no official validation",
            "training":{"from_scratch":True,"data":"train+fresh synthetic only",
                        "generator_sha256":CFG["generator_sha256"]},
        },f)
'''
source=source[:i]+replacement+source[j:]
ast.parse(source)
out=folder/"train_scratch_sf200_v2.py"
with out.open("x",encoding="utf-8",newline="\n") as f:f.write(source)
print("CREATED",str(out),"LINES",len(source.splitlines()))
