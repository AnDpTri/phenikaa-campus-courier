"""Unified V2 + role-semantic hard examples, built ON A BYTE-EXACT V2 CLONE.

API: generate(count, seed=0, holdout=False, mode='v2'|'hard'|'mixed').
mode='v2' is identical to V2's generate, including seeded examples.
The additional hard mode uses V18-style semantic *roles*, not copies of V18 rows.
No changes are made to the original D:\\phenikaa\\src package.
"""
from __future__ import annotations
import importlib.util
import random
import sys
from pathlib import Path

from courier.nlp.parser import TargetSpec
from courier.nlp.text import fold,tokenize

_THIS = Path(__file__).resolve()
_NAME = "courier.nlp._cloned_v2_unified_20261009"
if _NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_NAME, _THIS.with_name("synth_v2_clone.py"))
    _m = importlib.util.module_from_spec(_spec)
    sys.modules[_NAME] = _m
    _spec.loader.exec_module(_m)
V2 = sys.modules[_NAME]
Example = V2.Example
TYPES = V2.TYPES
GOAL_MODES = V2.GOAL_MODES
VIA_MODES = V2.VIA_MODES
V2Generator = V2.Generator

# Accented counterparts to V2's large (unaccented) lexical bank.
# Each bank keeps 1/5 of phrases out of training, just as V2 does.
ALIASES = {
    "library": ("thư viện trung tâm","phòng đọc sách","kho giáo trình","khu học liệu","quầy mượn sách","phòng tự học","kho tài liệu","thư viện khoa"),
    "dorm": ("ký túc xá sinh viên","khu nội trú","tòa ký túc xá","nhà ở sinh viên","khu lưu trú","dãy nhà sinh viên","tòa nhà ở nội trú","khu nhà ở sinh viên"),
    "sports": ("nhà thi đấu","sân bóng rổ","khu thể chất","sân tập thể thao","trung tâm thể thao","sân vận động","sân cầu lông","khu luyện tập"),
    "clinic": ("trạm y tế","phòng khám trong trường","phòng y tế","khu sơ cứu","trung tâm y tế","khu chăm sóc sức khỏe","phòng cấp cứu","trạm chăm sóc sinh viên"),
    "canteen": ("nhà ăn sinh viên","căn tin","quầy cơm","khu ẩm thực","nhà ăn trung tâm","quầy ăn trưa","khu phục vụ bữa trưa","khu ăn uống"),
    "parking": ("bãi gửi xe","nhà để xe","bãi đỗ xe","khu gửi xe sinh viên","bãi xe đạp","khu để xe máy","bãi giữ xe sinh viên","nhà xe"),
    "lecture": ("giảng đường chính","khu lớp học","tòa giảng đường","phòng học lý thuyết","tòa nhà học tập","khu giảng dạy","giảng đường lớn","nhà học"),
    "lab": ("phòng thí nghiệm","xưởng thực hành","khu nghiên cứu","phòng lab","trung tâm thực nghiệm","xưởng kỹ thuật","phòng thực hành","khu thí nghiệm"),
    "office": ("văn phòng khoa","phòng đào tạo","tòa hiệu bộ","phòng hành chính","khu văn phòng","phòng công tác sinh viên","văn phòng nhà trường","phòng quản lý"),
    "gate": ("cổng chính","cổng phụ","lối ra vào","chốt bảo vệ","trạm gác cổng","cổng trường","khu kiểm soát ra vào","cổng sau"),
}
assert set(ALIASES) == set(TYPES)
GENERIC_ITEMS = ("bưu kiện","gói hàng","thùng hàng","hộp hàng","túi hàng","kiện đồ","hộp carton","phong bì","túi bưu phẩm","thùng bưu phẩm")
PICKUP_ITEMS = ("bộ tài liệu phụ","phiếu giao hàng","tập biểu mẫu","phong bì chứng từ","túi giấy tờ","thẻ ra vào","tập giấy bổ sung","hộp chứng từ")
GREETING = ("Robot xử lý chuyến giao này giúp mình.","Có yêu cầu vận chuyển mới.","Nhờ robot đọc thông tin cập nhật.","Tôi gửi phiếu giao hàng mới.","Lệnh chuyển hàng vừa được xác nhận.","Có một kiện cần giao.","Mình nhờ robot chuyển kiện.","Yêu cầu của người gửi như sau.")
GOAL_ACTION = (
    "Giao {item} tới {loc}.",
    "Đưa {item} đến {loc}.",
    "Chuyển {item} sang {loc} để bàn giao.",
    "Địa chỉ giao {item} hiện tại là {loc}.",
    "Người nhận chờ {item} tại {loc}.",
    "Đơn này giao {item} đến {loc}.",
    "Điểm bàn giao cuối cùng cho {item} là {loc}.",
    "Nhờ robot mang {item} đến {loc}.",
    "Chặng giao cuối cùng là chuyển {item} đến {loc}.",
    "Hãy bàn giao {item} ở {loc}.",
)
VIA_ACTION = (
    "Robot nhận {pick} tại {loc}, sau đó thực hiện chặng giao.",
    "Cần nhận {pick} tại địa điểm sau: {loc}. Lấy xong mới thực hiện chặng giao.",
    "Điểm lấy bổ sung {pick} là địa chỉ sau: {loc}. Robot cần nhận hàng ở đó.",
    "Nhận {pick} từ {loc} rồi chuyển hàng đến đích đã xác nhận.",
    "Lộ trình có điểm nhận {pick} ở {loc} trước khi giao.",
    "Chặng nhận {pick} có địa chỉ: {loc}. Đích giao cuối cùng được nêu riêng.",
    "Điểm nhận kiện trung gian là {loc}, nơi robot lấy {pick}.",
    "Địa chỉ nhận {pick} như sau: {loc}. Robot lấy hàng tại điểm đó rồi đi giao.",
    "Đón {pick} từ {loc}; công đoạn sau là chuyển hàng tới đích.",
    "Chặng đầu robot cần tới {loc} thu {pick}, rồi tiếp tục giao hàng.",
    "Robot vòng qua {loc} để lấy {pick}, sau đó đến địa chỉ nhận cuối.",
    "Thêm bước nhận {pick} tại {loc}; không bỏ qua công đoạn này.",
)
# This bank has NO requirement to use word 'trước': corrects V18/V4 shortcut.
NO_VIA = (
    "Đã nhận đủ hàng tại nơi xuất phát, không có chặng lấy hàng trung gian.",
    "Người gửi đã giao {item} ngay tại điểm xuất phát, cứ đi thẳng tới đích.",
    "Lấy {item} tại nơi khởi hành, không ghé lấy thêm ở nơi khác.",
    "Chuyến này không yêu cầu một điểm lấy hàng phụ.",
    "Hàng đã được nhận từ đầu, robot chỉ phải giao đúng đích.",
    "Việc nhận {item} đã hoàn tất tại điểm xuất phát.",
    "Đi thẳng đến địa chỉ nhận; không có công đoạn ghé lấy bổ sung.",
    "Robot mang sẵn {item} khi bắt đầu nên không cần ghé kho khác.",
    "Chỉ có chặng giao, không có địa điểm nhận kiện trung gian.",
    "Không bố trí điểm gom thêm hàng trên tuyến đường.",
)
CANCEL_GOAL = (
    "Ban đầu phiếu ghi giao tới {old}, nhưng địa chỉ đó đã bị hủy.",
    "Đính chính: nơi giao {old} thuộc lệnh cũ, không còn hiệu lực.",
    "Lệnh trước yêu cầu giao ở {old}; yêu cầu ấy vừa được thay thế.",
    "Chỉ dẫn giao tới {old} trong bản nháp là sai; bỏ địa chỉ đó.",
    "Người gửi đã rút lại chỉ dẫn chuyển tới {old}.",
    "Trước đó người nhận chọn {old}, nhưng hiện đã thay nơi giao.",
    "Đừng thực hiện lệnh giao tới {old}; thông tin sau mới đúng.",
    "Chặng giao đến {old} bị xóa khỏi phiếu; dùng đích mới.",
    "Nơi nhận cũ được ghi tại {old}, nhưng giờ không dùng nữa.",
    "Cập nhật: bỏ nơi giao {old}, thay bằng chỉ dẫn tiếp theo.",
)
CANCEL_PICKUP = (
    "Ban đầu có điểm lấy hàng ở {old}; chặng này đã bị hủy.",
    "Không ghé {old} như phiếu lấy hàng cũ đã ghi.",
    "Đã rút lại yêu cầu nhận kiện tại {old}.",
    "Điểm lấy cũ {old} không còn hiệu lực.",
    "Phiếu trước đòi lấy hàng ở {old}; bây giờ đã bỏ chặng ấy.",
    "Người gửi hủy bước nhận hàng tại {old}.",
    "Đừng nhận hàng ở {old}; đó là điểm đã được thay thế.",
    "Đính chính: không phải lấy hàng tại {old}.",
    "Yêu cầu ghé {old} trước đây đã bị xóa.",
    "Tại {old} không còn kiện nào thuộc đơn này; hủy bước lấy.",
)
DISTRACTOR = {
    "history":("Hôm qua có đơn khác tới {loc}; không liên quan đơn hiện tại.","Chuyến cũ robot đã đi qua {loc}; bỏ qua thông tin đó.","Ghi chú về {loc} là lịch sử của một chuyến trước.","Địa chỉ {loc} thuộc khách hàng trước, không dùng cho đơn này.","Một phiếu cũ từng ghi {loc}, nhưng đơn hiện tại không liên quan."),
    "avoid":("Không giao tới {loc}; đó không phải đích hiện tại.","Đừng chuyển hàng nhầm sang {loc}.","Địa chỉ {loc} đã bị loại khỏi danh sách nơi giao.","Không chọn {loc} làm địa chỉ nhận của đơn này.","Thông tin {loc} chỉ là nội dung loại trừ, không phải nơi bàn giao."),
    "no_stop":("Đi ngang {loc} cũng không dừng lấy hàng.","Không cần ghé {loc} để nhận hoặc giao kiện.","Robot bỏ qua điểm dừng {loc}.","Đơn này không yêu cầu nhận hàng tại {loc}.","Chỉ đi qua {loc}, không được coi là điểm lấy trung gian."),
    "closed":("Bộ phận tại {loc} hôm nay đóng cửa, đừng giao tới đó.","Bộ phận tại {loc} không tiếp nhận hàng trong chuyến này.","Thông tin {loc} tạm ngừng tiếp nhận đã được ghi chú.","Khu vực tại {loc} đang ngừng hoạt động, không đến giao.","Đơn hiện tại không sử dụng {loc} vì nơi đó đóng cửa."),
    "moved":("Người nhận đã chuyển khỏi {loc}; đây là địa chỉ cũ.","Chỗ làm trước của người nhận là {loc}, không phải đích mới.","Người nhận không còn ở {loc}.","Thông tin người nhận tại {loc} đã hết hiệu lực.","Bản cũ ghi nơi ở {loc}; đừng suy ra đây là nơi giao."),
    "unrelated":("Thông tin về {loc} không liên quan đến chuyến này.","Có người nhắc {loc} trong câu chuyện khác, bỏ qua.","Địa danh {loc} chỉ là dữ kiện ngoài đơn giao.","Một thông báo riêng nói về {loc}, không phải lộ trình.","Chuyến hiện tại không liên quan tới địa chỉ {loc}."),
}
URGENT_POS=("Hạn giao đã được đẩy sớm, cần giao khẩn.","Đơn này phải ưu tiên xử lý ngay.","Ban đầu không vội, nay đổi thành giao hỏa tốc.","Lệnh mới xác nhận phải giao gấp.","Không được trì hoãn; đây là đơn khẩn cấp.","Người gửi vừa yêu cầu hoàn tất ngay.")
URGENT_NEG=("Đơn này không gấp.","Phiếu trước ghi hỏa tốc nhưng đã hủy ưu tiên.","Bây giờ có thể giao theo lịch thường.","Không cần ưu tiên xử lý ngay.","Thời hạn được nới, không cần vội.","Người gửi xác nhận có thể giao từ từ.")
FRAGILE_POS=("Kiện này chứa vật dễ vỡ, cần chống va đập.","Bên trong có món dễ bể; vận chuyển nhẹ tay.","Bản cũ ghi hàng bền, nhưng vừa đính chính là dễ vỡ.","Hàng đang chuyển rất mong manh, không được làm rơi.","Lô hàng nhạy va chạm, phải tránh rung lắc.","Đồ bên trong dễ hỏng nếu va đập, cần cẩn thận.")
FRAGILE_NEG=("Đơn này không chứa vật dễ vỡ.","Ban đầu ghi dễ vỡ nhưng kiểm tra lại là hàng bền.","Không cần chế độ vận chuyển đồ mong manh.","Kiện được xác nhận không có vật dễ bể.","Hàng này không nhạy va đập.","Lệnh mới bỏ yêu cầu chống va đập đặc biệt.")

def _eligible(bank, holdout):
    part = tuple(x for i,x in enumerate(bank) if (i%5==4)==holdout)
    return part or bank

class Generator(V2Generator):
    """One integrated V2 generator, with selectable difficulty.
    v2 = exact original; hard = role-aware V18 semantics; mixed = selectable ratio.
    """
    def __init__(self,seed:int=0,holdout:bool=False,mode:str="v2",hard_ratio:float=1/6):
        super().__init__(seed,holdout)
        if mode not in ("v2","hard","mixed"):raise ValueError(mode)
        if not 0<=hard_ratio<=1:raise ValueError(hard_ratio)
        self.mode=mode
        self.hard_ratio=hard_ratio
        self.last_provenance=None

    def _phrase(self, bank):
        return self.rng.choice(_eligible(bank,self.holdout))

    def _alias(self,kind,accented):
        return self._phrase(ALIASES[kind]) if accented else self.place(kind)

    def _choose_type(self,used):
        available=[t for t in TYPES if t not in used]
        if not available:raise ValueError("No available place type")
        return self.rng.choice(available)

    def _target(self,mode,used,accented):
        """Return target object and type-safe, self-contained nominal location phrase."""
        r=self.rng
        used=set(used)
        if mode.endswith("_most"):
            direction={"north_most":"bắc","south_most":"nam","east_most":"đông","west_most":"tây"}[mode]
            phrase=self._phrase((
              f"địa điểm ngoài cùng về phía {direction} trên bản đồ",
              f"địa danh nằm xa nhất về hướng {direction} của toàn khuôn viên",
              f"địa điểm ở cực {direction} của bản đồ",
              f"địa danh ở rìa {direction} xa nhất của sơ đồ",
              f"vị trí ở phía {direction} xa nhất trên sơ đồ",
              f"địa điểm tại mép {direction} của khuôn viên",
              f"vị trí ở phía {direction} ngoài cùng trên bản đồ",
              f"địa danh xa nhất về phía {direction} trong trường",
              f"điểm nằm tận cùng phía {direction} trong bản đồ",
              f"địa danh ở đầu {direction} của sơ đồ toàn trường",
            ))
            return TargetSpec(None,mode,None),phrase,used
        if mode=="anchor_near":
            anchor=self._choose_type(used);used.add(anchor)
            place=self._alias(anchor,accented)
            phrase=self._phrase((
                f"địa điểm gần {place} nhất, không tính chính {place}",
                f"vị trí cách {place} ít ô lưới nhất, loại trừ chính {place}",
                f"địa danh gần {place} nhất trên bản đồ, trừ mốc {place}",
                f"điểm có khoảng cách ngắn nhất tới {place}, không tính chính {place}",
                f"địa danh gần {place} hơn tất cả các vị trí khác, không kể {place}",
                f"điểm sát {place} nhất, không tính chính {place}",
                f"địa điểm cách {place} ngắn nhất, không tính chính {place}",
                f"điểm gần {place} nhất trong các điểm còn lại, không tính chính {place}",
                f"vị trí có cự ly tới {place} ngắn nhất, không tính chính {place}",
                f"địa danh gần {place} nhất, không tính chính {place}",
            ))
            return TargetSpec(None,mode,anchor),phrase,used
        kind=self._choose_type(used);used.add(kind)
        place=self._alias(kind,accented)
        if mode=="named":return TargetSpec(kind,None,None),place,used
        if mode in ("north","south","east","west"):
            w={"north":"bắc","south":"nam","east":"đông","west":"tây"}[mode]
            phrase=self._phrase((
                f"{place} ở phía {w} nhất trong các nơi cùng loại",
                f"{place} nằm ngoài cùng phía {w} của nhóm địa điểm cùng loại",
                f"{place} nằm xa hơn mọi nơi cùng loại về hướng {w}",
                f"{place} ở cực {w} trong nhóm cùng loại",
                f"{place} ở đầu {w} của các địa điểm cùng loại",
                f"{place} nằm sát mép {w} nhất trong nhóm cùng loại",
                f"{place} xa nhất về phía {w} so với các nơi cùng loại",
                f"{place} ở vị trí lệch về {w} nhất trong nhóm cùng loại",
                f"{place} nằm phía {w} hơn mọi nơi cùng loại",
                f"{place} là nơi cùng loại nằm xa nhất về phía {w}",
            ))
            return TargetSpec(kind,mode,None),phrase,used
        if mode in ("near","far"):
            anchor=self._choose_type(used);used.add(anchor)
            a=self._alias(anchor,accented)
            if mode=="near":
                phrase=self._phrase((
                    f"{place} gần {a} nhất trong nhóm cùng loại",
                    f"{place} gần {a} hơn các nơi cùng loại còn lại",
                    f"{place} có khoảng cách tới {a} ngắn nhất trong nhóm cùng loại",
                    f"{place} sát {a} hơn những nơi cùng loại khác",
                    f"{place} có cự ly ngắn nhất đến {a} trong nhóm cùng loại",
                    f"{place} là nơi cùng loại gần {a} nhất",
                    f"{place} nằm gần {a} nhất trong các lựa chọn cùng loại",
                    f"{place} có vị trí gần {a} hơn địa điểm tương tự còn lại",
                    f"{place} ở phía gần {a} hơn những nơi cùng loại",
                    f"{place} nằm cách {a} ít ô hơn các nơi cùng loại khác",
                ))
            else:
                phrase=self._phrase((
                    f"{place} xa {a} nhất trong nhóm cùng loại",
                    f"{place} xa {a} hơn các nơi cùng loại còn lại",
                    f"{place} có khoảng cách tới {a} dài nhất trong nhóm cùng loại",
                    f"{place} xa {a} hơn những nơi tương tự khác",
                    f"{place} có cự ly lớn nhất đến {a} trong nhóm cùng loại",
                    f"{place} là nơi cùng loại xa {a} nhất",
                    f"{place} nằm xa {a} nhất trong các lựa chọn cùng loại",
                    f"{place} có vị trí xa {a} hơn địa điểm tương tự còn lại",
                    f"{place} ở phía xa {a} hơn những nơi cùng loại",
                    f"{place} nằm cách {a} nhiều ô hơn các nơi cùng loại khác",
                ))
            return TargetSpec(kind,mode,anchor),phrase,used
        raise ValueError(mode)

    def _hard_example(self):
        r=self.rng
        accented=r.random()<.60
        gm=r.choices(["named","near","far","anchor_near","north","south","east","west",
                      "north_most","south_most","east_most","west_most"],
                     [34,14,10,14,4,4,4,4,3,3,3,3])[0]
        vm=r.choices(["none","named","near","far","north","south","east","west"],
                     [49,34,5,4,2,2,2,2])[0]
        goal,gtext,used=self._target(gm,(),accented)
        via,vtext=(None,None)
        if vm!="none":via,vtext,used=self._target(vm,used,accented)
        item=self._phrase(GENERIC_ITEMS)
        pickitem=self._phrase(PICKUP_ITEMS) if via is not None and r.random()<.50 else item
        urgent=r.random()<.46
        fragile=r.random()<.45
        core_goal=self._phrase(GOAL_ACTION).format(item=item,loc=gtext)
        core_via=(self._phrase(VIA_ACTION).format(pick=pickitem,loc=vtext)
                  if via is not None else self._phrase(NO_VIA).format(item=item))
        clauses=[]
        if r.random()<.26:clauses.append(self._phrase(GREETING))
        former_goal=None
        if goal.type is not None and r.random()<.32:
            old_type=self._choose_type(used);used.add(old_type)
            former_goal=self._alias(old_type,accented)
            clauses.append(self._phrase(CANCEL_GOAL).format(old=former_goal))
        former_pickup=None
        if goal.type is not None and r.random()<.20:
            old_type=self._choose_type(used);used.add(old_type)
            former_pickup=self._alias(old_type,accented)
            clauses.append(self._phrase(CANCEL_PICKUP).format(old=former_pickup))
        if r.random()<.5:clauses.extend((core_via,core_goal))
        else:clauses.extend((core_goal,core_via))
        neg=[]
        for _ in range(r.choices([0,1,2],[.36,.57,.07])[0]):
            if len(used)>=len(TYPES):break
            t=self._choose_type(used);used.add(t)
            role=r.choice(("history","unrelated") if goal.type is None else tuple(DISTRACTOR))
            place=self._alias(t,accented)
            neg.append((role,t,place))
            clauses.append(self._phrase(DISTRACTOR[role]).format(loc=place))
        # Positive flags MUST be explicitly grounded. Negative may be omitted.
        if urgent or r.random()<.45:clauses.append(self._phrase(URGENT_POS if urgent else URGENT_NEG))
        if fragile or r.random()<.45:clauses.append(self._phrase(FRAGILE_POS if fragile else FRAGILE_NEG))
        text=" ".join(clauses)
        if not accented:text=fold(text)
        from courier.nlp.neural import MAX_TOKENS,spec_labels
        if not (14<=len(tokenize(text))<=MAX_TOKENS):
            return None
        if len(spec_labels(goal,via,urgent,fragile))!=8:raise ValueError("Bad target labels")
        self.last_provenance={
            "mode":"hard","former_destination":former_goal,"former_pickup":former_pickup,
            "negative_roles":[{"role":role,"type":t,"text":p} for role,t,p in neg],
            "goal_mode":gm,"via_mode":vm,"item":item,"pickup_item":pickitem,
            "accented":accented,
        }
        return Example(text,goal,via,urgent,fragile)

    def example(self):
        chosen=self.mode
        if chosen=="mixed":chosen="hard" if self.rng.random()<self.hard_ratio else "v2"
        if chosen=="v2":
            ex=super().example()
            self.last_provenance={"mode":"v2"}
            return ex
        for _ in range(60):
            ex=self._hard_example()
            if ex is not None:return ex
        raise RuntimeError("Hard example rejection limit")

def generate(count:int,seed:int=0,holdout:bool=False,mode:str="v2",hard_ratio:float=1/6):
    gen=Generator(seed,holdout,mode,hard_ratio)
    return [gen.example() for _ in range(count)]

def generate_roles(count:int,seed:int=0,holdout:bool=False,mode:str="hard",hard_ratio:float=1/6):
    """Return (Example, provenance) preserving resolved role labels for audit."""
    gen=Generator(seed,holdout,mode,hard_ratio)
    out=[]
    for _ in range(count):
        ex=gen.example()
        out.append((ex,gen.last_provenance))
    return out

# Clone's functions remain available for annotated real augmentation.
augment_real=V2.augment_real
spec_from_mission=V2.spec_from_mission
