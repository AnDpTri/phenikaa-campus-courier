"""V11 new 40k, apply audited V9/V10 patches from immutable scripts via AST, then small V11 fixes.
No file is overwritten or modified.
"""
import ast
from pathlib import Path
P=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v8_20261009_first")
src=(P/"typed_semantic_generator.py").read_text(encoding="utf-8")
def get_literals(script_name,var):
 tree=ast.parse((P/script_name).read_text(encoding="utf-8"))
 for node in tree.body:
  if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id==var:
   return ast.literal_eval(node.value)
 raise RuntimeError(f"No list: {script_name} {var}")
changes=get_literals("run_typed_semantic_v9.py","changes")
changes+=get_literals("run_typed_semantic_v10.py","extra")
changes+=[
 ('N=2000','N=40000'),
 ('SEED=2026100920','SEED=2026100921'),
 ('OUTPUT=BASE/"pilot_v10_2000.jsonl"','OUTPUT=BASE/"specialized_new_40000_v11.jsonl"'),
 ('META=BASE/"pilot_v10_stats.json"','META=BASE/"specialized_new_40000_v11_stats.json"'),
 ('f"sem_v8_{i:06d}"','f"sem_v11_{i:06d}"'),
 ('ct=rng.choices([0,1,2],weights=[.40,.57,.03])[0]','ct=rng.choices([0,1,2],weights=[.33,.63,.04])[0]'),
 ('"hộp lọ thuốc thử bằng kính"','"bộ lọ thuốc thử bằng thủy tinh"'),
]
for a,b in changes:
 assert src.count(a)==1,(a,src.count(a))
 src=src.replace(a,b)
ctx={"__name__":"v11_new_40k_compiled","__file__":str(P/"typed_semantic_generator.py")}
exec(compile(src,str(P/"typed_semantic_generator.py"),"exec"),ctx)
ctx["main"]()
