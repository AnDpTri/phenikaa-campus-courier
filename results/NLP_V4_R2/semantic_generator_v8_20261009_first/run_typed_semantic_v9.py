"""V9 in-memory, create-only continuation of typed semantic generation; source V8 immutable."""
from pathlib import Path
P=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v8_20261009_first")
src=(P/"typed_semantic_generator.py").read_text(encoding="utf-8")
changes=[
('SEED=2026100918','SEED=2026100919'),
('OUTPUT=BASE/"pilot_v8_2000.jsonl"','OUTPUT=BASE/"pilot_v9_2000.jsonl"'),
('META=BASE/"pilot_v8_stats.json"','META=BASE/"pilot_v9_stats.json"'),
('"Bưu kiện {item} được giữ tại {loc}; hãy ghé lấy trước khi đi đến đích."','"{item} đang được giữ tại {loc}; hãy tới nhận rồi đi đến đích."'),
('f"địa danh khác {landmark} gần mốc {landmark} nhất theo khoảng cách ô lưới"','f"địa danh gần {landmark} nhất tính theo ô lưới, ngoài chính {landmark}"'),
('f"địa danh gần {landmark} nhất tính theo số ô lưới, không tính chính {landmark}"','f"địa điểm gần {landmark} nhất theo ô lưới, trừ chính {landmark}"'),
('f"địa điểm có khoảng cách ô lưới ngắn nhất tới {landmark}, ngoại trừ chính {landmark}"','f"vị trí gần {landmark} nhất theo ô lưới, không tính mốc {landmark}"'),
('f"{name} ở phía {axis} nhất trong các địa điểm cùng loại trên bản đồ"','f"{name} ở phía {axis} nhất trong các nơi cùng loại"'),
('f"{name} nằm xa nhất về phía {axis} so với các địa điểm cùng loại"','f"{name} nằm ngoài cùng phía {axis} trong nhóm địa điểm cùng loại"'),
('f"{name} thuộc vị trí ngoài cùng phía {axis} trong số các địa điểm cùng loại"','f"{name} xa nhất về hướng {axis} so với các nơi cùng loại"'),
('f"{name} {words[0]} {anchor} nhất trong các địa điểm cùng loại, tính theo đường chim bay"','f"{name} {words[0]} {anchor} nhất trong các nơi cùng loại"'),
('f"{name} có khoảng cách đường chim bay {words[1]} tới {anchor} trong số các địa điểm cùng loại"','f"{name} có khoảng cách đường chim bay {words[1]} đến {anchor} trong cùng loại"'),
('f"{name} {words[0]} {anchor} hơn các địa điểm cùng loại còn lại"','f"{name} {words[0]} {anchor} hơn các nơi cùng loại còn lại"'),
('ct=rng.choices([0,1,2],weights=[.30,.55,.15])[0]','ct=rng.choices([0,1,2],weights=[.35,.59,.06])[0]'),
('if rng.random()<.86 or urgent:','if rng.random()<.64 or urgent:'),
('if rng.random()<.84 or fragile:','if rng.random()<.64 or fragile:'),
('if rng.random()<.20:blocks.append(choice(NEUTRAL))','if rng.random()<.09:blocks.append(choice(NEUTRAL))'),
('if rng.random()<.53:blocks.insert(0,choice(GREET))','if rng.random()<.34:blocks.insert(0,choice(GREET))'),
]
for a,b in changes:
 if src.count(a)!=1:raise ValueError((a,src.count(a)))
 src=src.replace(a,b)
ctx={"__name__":"v9_inmemory","__file__":str(P/"typed_semantic_generator.py")}
exec(compile(src,str(P/"typed_semantic_generator.py"),"exec"),ctx)
ctx["main"]()
