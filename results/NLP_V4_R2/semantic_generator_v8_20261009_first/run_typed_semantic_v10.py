"""V10 typed generator, trained-code untouched, quality fixes via in-memory rewrite."""
from pathlib import Path
P=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v8_20261009_first")
w=(P/"run_typed_semantic_v9.py").read_text(encoding="utf-8")
assert w.count('2026100919')==1
w=w.replace('2026100919','2026100920')
assert w.count('pilot_v9_2000.jsonl')==1
w=w.replace('pilot_v9_2000.jsonl','pilot_v10_2000.jsonl')
assert w.count('pilot_v9_stats.json')==1
w=w.replace('pilot_v9_stats.json','pilot_v10_stats.json')
i=w.index(']\nfor a,b in changes:')
extra=[
 ('ct=rng.choices([0,1,2],weights=[.35,.59,.06])[0]','ct=rng.choices([0,1,2],weights=[.40,.57,.03])[0]'),
 ('if rng.random()<.64 or urgent:','if rng.random()<.57 or urgent:'),
 ('if rng.random()<.64 or fragile:','if rng.random()<.57 or fragile:'),
 ('if rng.random()<.34:blocks.insert(0,choice(GREET))','if rng.random()<.27:blocks.insert(0,choice(GREET))'),
 ('f"{name} xa nhất về hướng {axis} so với các nơi cùng loại"','f"{name} nằm ở cực {axis} trong số các nơi cùng loại"'),
 ('blocks.append(choice(CANCEL).format(decoy=decoy))',
 'blocks.append(choice(("Trong phiếu chuyến trước có nhắc tới {decoy}; đây là đơn khác.","Hôm qua robot đã đi qua {decoy} trong một chuyến khác.","Ghi chú cũ có nhắc {decoy}; ghi chú này đã hết hiệu lực.")).format(decoy=decoy) if goal.type is None else choice(CANCEL).format(decoy=decoy))'),
]
ins="".join(' ('+repr(a)+','+repr(b)+'),\n' for a,b in extra)
w=w[:i]+ins+w[i:]
ctx={"__name__":"typed_generator_v10_inmemory","__file__":str(P/"run_typed_semantic_v9.py")}
exec(compile(w,str(P/"run_typed_semantic_v9.py"),"exec"),ctx)
