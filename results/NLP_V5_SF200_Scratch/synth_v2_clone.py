"""Synthetic labelled missions for training the neural mission parser.

Every phrase below is hand-written general Vietnamese (accent-folded, like
`text.fold`) or taken from the lexicon, which is itself built from train only.
Nothing is taken from test.

Each phrase bank is split deterministically into a "train" part and a
"holdout" part (every fifth phrase). `generate(..., holdout=True)` builds the
core of every mission (place names, spatial phrasing, verbs, via cues) from
holdout phrases only, so it measures how well a parser handles phrasing it has
never seen. Training data uses the train part only.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from . import lexicon as lx
from .parser import TargetSpec
from .text import fold

TYPES = tuple(lx.PLACE_ALIASES)
GOAL_MODES = (
    "named", "north", "south", "west", "east", "near", "far",
    "north_most", "south_most", "west_most", "east_most", "anchor_near",
)
VIA_MODES = ("none", "named", "north", "south", "west", "east", "near", "far")
DIRECTIONS = ("north", "south", "west", "east")


@dataclass(frozen=True, slots=True)
class Example:
    text: str
    goal: TargetSpec
    via: TargetSpec | None
    urgent: bool
    fragile: bool


# ---------------------------------------------------------------- phrase banks

EXTRA_PLACES: dict[str, tuple[str, ...]] = {
    "library": ("khu thu vien", "toa thu vien", "phong muon sach", "thu vien trung tam", "kho sach", "phong tu hoc"),
    "dorm": ("khu ky tuc xa", "toa ktx", "nha ky tuc", "khu luu tru sinh vien", "day nha ktx", "noi o cua sinh vien"),
    "sports": ("san bong da", "san bong ro", "khu the duc", "nha tap the thao", "san co", "trung tam the duc"),
    "clinic": ("phong kham da khoa", "phong y te truong", "co so y te", "tram cuu thuong", "phong cap cuu", "y te truong"),
    "canteen": ("nha an sinh vien", "khu bep", "quan an truong", "phong an chung", "nha bep", "khu nha an"),
    "parking": ("bai giu xe", "khu de xe", "ham gui xe", "nha giu xe", "bai dau xe", "san de xe"),
    "lecture": ("toa giang day", "giang duong lon", "khu hoc tap", "phong hoc chung", "nha hoc", "toa lop hoc"),
    "lab": ("phong thi nghiem hoa", "trung tam thi nghiem", "xuong", "phong thi nghiem ly", "khu nghien cuu", "phong lab sinh"),
    "office": ("phong cong tac sinh vien", "van phong truong", "phong ke toan", "toa nha dieu hanh", "phong tiep dan", "van phong khoa"),
    "gate": ("cong truong chinh", "cong ra vao", "cong bao ve", "loi ra vao", "cong lon", "cong phia truoc"),
}

EXTRA_PERSONS: dict[str, tuple[str, ...]] = {
    "library": ("co thu thu", "nguoi giu sach", "nhan vien muon tra sach"),
    "dorm": ("tro ly ktx", "ban dieu hanh ktx", "sinh vien o ky tuc"),
    "sports": ("doi bong ro", "thay day the duc", "cau lac bo bong da"),
    "clinic": ("bac si truc", "y ta truc", "nhan vien phong kham"),
    "canteen": ("chu quan an", "nhan vien nha an", "co phu bep"),
    "parking": ("nguoi trong xe", "bac trong xe", "nhan vien bai xe"),
    "lecture": ("thay chu nhiem", "co giang vien", "lop dang hoc"),
    "lab": ("tro giang phong lab", "nhom thi nghiem", "can bo phong thi nghiem"),
    "office": ("co van thu", "can bo giao vu", "thay truong phong"),
    "gate": ("chu bao ve", "doi bao ve", "nguoi gac cong"),
}

EXTRA_ITEMS = (
    "goi hang", "thung hang", "buu kien", "hop giay", "tui do", "phong bi", "chiec hop", "kien hang",
    "mon qua", "bo tai lieu", "thung carton", "chiec tui", "lo hang", "don hang", "goi do", "cai hop",
    "bo ho so", "chong sach", "thung sua", "hop banh", "laptop", "o cung", "chai nuoc", "gio hoa qua",
)

DELIVER_VERBS = (
    "giao", "mang", "chuyen", "dua", "gui", "cho", "dem", "van chuyen", "chuyen phat", "ship",
    "phat", "mang giup", "giao giup", "chuyen giup", "dua giup", "chuyen ho", "giao ho", "mang ho",
)
TO_WORDS = ("toi", "den", "qua", "sang", "vao", "ra", "len", "xuong")
POLITE_TAILS = ("giup minh", "giup em", "giup toi", "nhe", "nha", "voi", "ho minh", "di", "nhe robot", "")

# Core goal sentences; {L} = location phrase, {I} = item, {V} = deliver verb, {T} = to word.
GOAL_FRAMES = (
    "{V} {I} {T} {L} {tail}",
    "{V} {I} {T} {L}",
    "nho robot {V} {I} {T} {L}",
    "co don giao {I} o {L}",
    "can {I} o {L}",
    "diem giao: {L}. hang: {I}",
    "dich den la {L}",
    "noi nhan: {L}",
    "hay di {T} {L} {tail}",
    "ban {V} {I} {T} {L} {tail}",
    "{I} can duoc {V} {T} {L}",
    "giao hang tai {L}",
    "nguoi nhan dang o {L}",
    "dia chi giao: {L}",
    "hang can toi {L}",
    "lam on {V} {I} {T} {L}",
    "phien robot {V} {I} {T} {L}",
    "chuyen nay di {L}",
    "{L} dang cho {I}",
    "toi can gui {I} {T} {L}",
    "minh muon {V} {I} {T} {L}",
    "giao {T} {L} {tail}",
    "dem {I} {T} {L} ngay",
    "dia diem nhan hang la {L}",
    "nho {V} {I} toi tan {L}",
    "kien hang nay di {L}",
    "robot chay {T} {L} giao {I}",
    "vui long {V} {I} {T} {L}",
    "{V} {T} {L} giup",
    "lay {I} roi {V} {T} {L}",
)
PERSON_FRAMES = (
    "{P} dang cho {I}",
    "{P} can {I}",
    "{P} o {L} dang cho {I}",
    "{V} {I} cho {P}",
    "{V} {I} cho {P} o {L}",
    "{P} nho gui {I}",
    "{P} dang doi {I} gap",
    "nguoi nhan la {P}",
)

# Spatial phrasing. {P}=place, {A}=anchor place, {G}=generic head noun.
DIR_WORDS = {
    "north": ("bac", "phia bac", "ben tren", "phia tren", "o tren", "nua tren", "mien bac", "huong bac", "tren"),
    "south": ("nam", "phia nam", "ben duoi", "phia duoi", "o duoi", "nua duoi", "mien nam", "huong nam", "duoi"),
    "west": ("tay", "phia tay", "ben trai", "phia trai", "o ben trai", "nua trai", "huong tay", "trai"),
    "east": ("dong", "phia dong", "ben phai", "phia phai", "o ben phai", "nua phai", "huong dong", "phai"),
}
DIR_FRAMES = (
    "{P} {D}",
    "{P} nam {D}",
    "{P} o {D}",
    "{P} o {D} ban do",
    "{P} nam o {D} ban do",
    "{P} phia {d}",
    "cai {P} o {D}",
    "{P} (cai o {D})",
    "{P} thuoc {D} ban do",
    "trong hai {P}, cai o {D}",
    "{P} ve {D}",
    "{P} khu {d}",
)
NEAR_FRAMES = (
    "{P} gan {A} hon",
    "{P} nam gan {A}",
    "{P} o gan {A}",
    "{P} sat {A} hon",
    "{P} gan {A}",
    "{P} canh {A}",
    "{P} o canh {A}",
    "{P} cach {A} gan hon",
    "trong hai {P}, cai gan {A} hon",
    "{P} nam ke {A}",
    "{P} phia gan {A}",
    "{P} gan voi {A} hon",
    "{P} o sat {A}",
    "{P} lien ke {A}",
)
FAR_FRAMES = (
    "{P} xa {A} hon",
    "{P} nam xa {A}",
    "{P} o xa {A}",
    "{P} cach xa {A}",
    "trong hai {P}, cai xa {A} hon",
    "{P} xa {A}",
    "{P} phia xa {A}",
    "{P} cach {A} xa hon",
    "{P} o dau xa {A}",
    "{P} khong gan {A}",
)
GENERIC_HEADS = (
    "dia diem", "noi", "cho", "toa nha", "khu", "diem", "vi tri", "khu vuc", "toa", "dia chi", "o", "nut",
)
ANCHOR_NEAR_FRAMES = (
    "{G} nam sat {A} nhat",
    "{G} gan {A} nhat",
    "{G} canh {A} nhat",
    "{G} o gan {A} nhat",
    "{G} sat {A} nhat",
    "{G} gan voi {A} nhat",
    "{G} ke ben {A}",
    "{G} ngay canh {A}",
    "{G} lien ke {A}",
    "{G} cach {A} it buoc nhat",
    "{G} gan {A} nhat (khong tinh {A})",
    "{G} nao gan {A} nhat",
    "{G} nam ke {A} nhat",
    "{G} ngay sat {A}",
    "{G} lang gieng gan nhat cua {A}",
    "{G} cach {A} ngan nhat",
    "{G} it xa {A} nhat",
    "{G} o sat ben {A}",
    "{G} gan ke {A} nhat",
    "{G} thuoc lan can {A}",
    "{G} sat vach {A}",
    "{G} nam gan {A} nhat tren ban do",
    "{G} gan sat {A}",
    "{G} ke {A} nhat",
)
MOST_PHRASES = {
    "north_most": (
        "nam cao nhat tren ban do", "xa nhat ve phia bac", "o tren cung", "o tren cung ban do", "sat mep tren nhat",
        "cuc bac cua ban do", "o vi tri cao nhat", "nam o hang tren cung", "o hang dau tien", "bac nhat",
        "o phia bac xa nhat", "tren cung cua ban do", "o ria tren", "gan mep tren nhat", "o goc tren cung",
        "xa nhat ve phia tren", "nam tren cung", "o dinh ban do", "phia bac nhat", "nam sat bien tren nhat",
        "o dong tren cung", "cao nhat", "nam o phia tren nhat", "tren nhat",
    ),
    "south_most": (
        "nam thap nhat tren ban do", "xa nhat ve phia nam", "o duoi cung", "o duoi cung ban do", "sat mep duoi nhat",
        "cuc nam cua ban do", "o vi tri thap nhat", "nam o hang duoi cung", "o hang cuoi cung", "nam nhat",
        "o phia nam xa nhat", "duoi cung cua ban do", "o ria duoi", "gan mep duoi nhat", "o goc duoi cung",
        "xa nhat ve phia duoi", "nam duoi cung", "o day ban do", "phia nam nhat", "nam sat bien duoi nhat",
        "o dong duoi cung", "thap nhat", "nam o phia duoi nhat", "duoi nhat",
    ),
    "west_most": (
        "nam sat mep trai nhat", "xa nhat ve phia tay", "ngoai cung ben trai", "o ben trai cung", "sat le trai nhat",
        "cuc tay cua ban do", "o cot trai nhat", "nam o cot dau tien", "o cot ngoai cung ben trai", "tay nhat",
        "o phia tay xa nhat", "trai cung cua ban do", "o ria trai", "gan mep trai nhat", "o goc trai",
        "xa nhat ve ben trai", "nam trai cung", "phia tay nhat", "nam sat bien trai nhat", "trai nhat",
        "o phia trai nhat", "lech trai nhat", "nam ngoai cung phia tay", "o mep trai",
    ),
    "east_most": (
        "nam sat mep phai nhat", "xa nhat ve phia dong", "ngoai cung ben phai", "o ben phai cung", "sat le phai nhat",
        "cuc dong cua ban do", "o cot phai nhat", "nam o cot cuoi cung", "o cot ngoai cung ben phai", "dong nhat",
        "o phia dong xa nhat", "phai cung cua ban do", "o ria phai", "gan mep phai nhat", "o goc phai",
        "xa nhat ve ben phai", "nam phai cung", "phia dong nhat", "nam sat bien phai nhat", "phai nhat",
        "o phia phai nhat", "lech phai nhat", "nam ngoai cung phia dong", "o mep phai",
    ),
}
MOST_FRAMES = ("{G} {M}", "{G} {M}", "{G} nao {M}", "{G} ma {M}", "{M}")

VIA_FRAMES_BEFORE = (  # via sentence placed before the goal sentence
    "truoc tien qua {W} nhan {I2}, sau do",
    "truoc het ghe {W} lay {I2}, roi",
    "ghe {W} lay {I2} truoc, roi",
    "di qua {W} truoc, sau do",
    "lay {I2} o {W} xong thi",
    "sau khi lay {I2} o {W},",
    "dau tien toi {W} nhan {I2}, tiep theo",
    "tat qua {W} lay {I2} roi",
    "truoc tien vao {W}, xong roi",
    "buoc mot: ghe {W}. buoc hai:",
    "ban hay qua {W} truoc roi",
    "ghe {W} truoc da, sau do",
)
VIA_FRAMES_AFTER = (  # via sentence placed after the goal sentence
    "nho ghe {W} lay {I2} truoc",
    "nhung phai tat qua {W} truoc da",
    "tren duong di nho ghe {W}",
    "can ghe {W} truoc",
    "nho tat vao {W} lay {I2} truoc khi giao",
    "ghe qua {W} truoc nhe",
    "truoc do phai di qua {W}",
    "nho di ngang {W} lay {I2} truoc",
    "phai ghe {W} truoc roi moi giao",
    "nho ghe {W} lay {I2} truoc khi toi diem giao",
    "nho dung o {W} truoc",
    "truoc khi giao phai ghe {W}",
)

DISTRACTOR_FRAMES = (
    "dung nham voi {X} nhe",
    "khong can ghe {X}",
    "hom qua da giao o {X} roi",
    "nguoi nhan da roi {X} roi",
    "(tin truoc ghi {X} la nham)",
    "khong phai {X} dau",
    "{X} thi khong can toi",
    "dung di qua {X}",
    "tranh {X} ra nhe",
    "khong giao o {X} nua",
    "dia chi cu la {X}, bo qua",
    "{X} hom nay dong cua",
    "dung ghe {X}",
    "khong lien quan gi toi {X}",
    "khong can qua {X}",
    "dung de o {X}",
    "lan truoc giao nham {X}",
    "khong phai giao {X}",
)
REDIRECT_PREFIX = (
    "huy don giao {X}. thay vao do:",
    "luc nay nhan nham la {X}. dung ra:",
    "khong phai {X} ma la",
    "doi dia chi tu {X} sang",
    "bo don cu o {X}. don moi:",
    "sua lai: khong phai {X},",
    "nham roi, khong phai {X}.",
)

URGENT_TRUE = (
    "gap nhe!", "hoa toc!", "can ngay trong 5 phut.", "viec nay khan cap.", "dang can gap lam.",
    "cang nhanh cang tot.", "giao ngay nhe.", "can gap!", "khan cap!", "nhanh len nhe.",
    "uu tien giao truoc.", "dang rat gap.", "gap lam roi.", "lam nhanh giup minh.", "can ngay lap tuc.",
    "trong vong 10 phut nhe.", "som cang tot.", "toc hanh nhe.", "gap gap!", "giao lien nhe.",
)
URGENT_FALSE = (
    "khong gap dau.", "cu tu tu, khong voi.", "chieu nay giao cung duoc.", "khong can voi.", "tu tu cung duoc.",
    "mai giao cung duoc.", "khong gap lam.", "luc nao ranh thi giao.", "khong can gap.", "cham cung khong sao.",
    "khong vi phai nhanh.", "khong co gi gap.",
)
FRAGILE_TRUE = (
    "hang de vo, di can than.", "nhe tay vi do rat de vo.", "ben trong la do thuy tinh.", "do gom ben trong, tranh va cham.",
    "chu y: hang de hong khi va dap.", "hang de vo nhe.", "can than do su.", "ben trong co trung, nhe tay.",
    "hang de be, cam nhe.", "do thuy tinh, di cham thoi.", "hang mong manh lam.", "coi chung vo nhe.",
    "tranh rung lac vi de vo.", "dung lam roi, de vo lam.",
)
FRAGILE_FALSE = (
    "hang chac chan, khong de vo.", "do khong vo duoc dau.", "hang ben, khong lo vo.", "do cung, khong sao.",
    "khong de vo.", "hang khong vo duoc.", "khong can nhe tay.",
)
GREETINGS = (
    "xin chao!", "robot oi,", "nho ban nhe:", "yeu cau moi:", "chao robot!", "alo robot,", "hello robot,",
    "phien ban chut:", "chao ban,", "robot than men,", "[don #{n}]", "ma don {n}:", "",
)
CLOSINGS = ("cam on!", "cam on nhieu.", "thanks robot.", "cam on ban nhe.", "tks.", "nho nhe.", "")


# ---------------------------------------------------------------- expansion (v2)
# More general Vietnamese phrasing, appended so the train/holdout split of the
# phrases above is unchanged.

for _kind, _extra in {
    "library": ("trung tam hoc lieu", "phong doc tu chon", "kho tai lieu", "khu muon tra sach", "phong sach"),
    "dorm": ("nha luu tru", "khu nha o", "toa o sinh vien", "cho o noi tru", "khu ky tuc"),
    "sports": ("nha da nang", "san tennis", "khu tap luyen", "san thi dau", "cho tap the duc"),
    "clinic": ("phong cuu thuong", "tram y", "noi so cuu", "phong kham suc khoe", "khu kham benh"),
    "canteen": ("nha an chung", "khu an trua", "quay an", "phong an sinh vien", "can teen"),
    "parking": ("bai do", "cho gui xe may", "khu de xe dap", "nha de xe may", "bai xe sinh vien"),
    "lecture": ("giang duong chinh", "phong hoc ly thuyet", "khu day hoc", "toa hoc", "lop ly thuyet"),
    "lab": ("phong thuc tap", "phong thi nghiem sinh", "khu lab", "phong lam thi nghiem", "xuong co khi"),
    "office": ("phong quan ly dao tao", "van phong hanh chinh", "phong tong hop", "toa van phong", "phong cong vu"),
    "gate": ("cong ngoai", "cong ra", "phong bao ve", "loi ra", "cong rao"),
}.items():
    EXTRA_PLACES[_kind] += _extra

for _kind, _extra in {
    "library": ("ban quan ly thu vien", "nhan vien phong doc"),
    "dorm": ("ban tu quan ktx", "nguoi o ktx"),
    "sports": ("doi van dong vien", "ban to chuc giai the thao"),
    "clinic": ("y si", "can bo y te"),
    "canteen": ("chi nau bep", "quan ly nha an"),
    "parking": ("nguoi giu xe", "doi trong xe"),
    "lecture": ("lop dang thi", "thay day ly thuyet"),
    "lab": ("ban tro giang lab", "nguoi phu trach phong thi nghiem"),
    "office": ("phong ke hoach", "nhan vien hanh chinh"),
    "gate": ("ca truc bao ve", "nguoi gac"),
}.items():
    EXTRA_PERSONS[_kind] += _extra

EXTRA_ITEMS += (
    "phong thu", "chia khoa phong", "bang ten", "ao dong phuc", "hop dung cu", "tui rac", "bo sach",
    "giay moi", "the sinh vien", "ban photo", "may tinh bang", "tai nghe", "loa", "bo sac", "cuc pin",
)

GOAL_FRAMES += (
    "{L} la noi can giao",
    "noi giao hang: {L}",
    "diem den cua don nay la {L}",
    "giao toi {L} nhe, do la {I}",
    "chuyen phat {I} den {L} gap",
    "{I} gui {T} {L}",
    "hang di {L}",
    "can dua {I} {T} {L}",
    "co {I} can giao, dia chi la {L}",
    "giao {I}. dia chi: {L}",
    "ve {L} giao {I}",
    "ban oi, {V} {I} {T} {L} nha",
    "muc tieu: {L}",
    "den {L} nha robot",
    "toi {L} de giao {I}",
    "nhan hang tai {L}",
    "{I} nay can co mat o {L}",
    "xin {V} {I} {T} {L}",
    "giao cho nguoi o {L}",
    "nguoi nhan cho o {L}",
)
PERSON_FRAMES += (
    "giao cho {P} nhe",
    "{P} se nhan {I}",
    "{I} la cua {P}",
    "{P} vua dat {I}",
    "mang {I} toi cho {P}",
)
DIR_FRAMES += (
    "{P} ben {d}",
    "{P} canh {d}",
    "{P} o goc {d}",
    "{P} lech {d}",
    "{P} thuoc khu vuc {D}",
)
NEAR_FRAMES += (
    "{P} o gan {A} hon cai kia",
    "{P} ngay gan {A}",
    "{P} gan phia {A}",
    "{P} sat ben {A}",
    "{P} cai ma gan {A}",
)
FAR_FRAMES += (
    "{P} xa {A} hon cai kia",
    "{P} nam cach xa {A}",
    "{P} o xa phia {A}",
    "{P} cai ma xa {A}",
    "{P} khong o gan {A}",
)
GENERIC_HEADS += ("can phong", "toa nao", "dia danh", "dia diem nao do", "cong trinh", "tram")
ANCHOR_NEAR_FRAMES += (
    "{G} ke canh {A}",
    "{G} sat canh {A} nhat",
    "{G} nam gan {A} nhat",
    "{G} gan {A} hon tat ca",
    "{G} o canh {A} nhat",
    "{G} cach {A} gan nhat",
    "{G} gan nhat voi {A}",
    "{G} lien ke voi {A} nhat",
    "{G} gan {A} nhat tren so do",
    "{G} sat {A}",
    "{G} ngay ben canh {A}",
    "{G} nao nam sat {A} nhat",
    "{G} it buoc nhat tu {A}",
    "{G} di bo tu {A} gan nhat",
    "{G} hang xom cua {A}",
)
for _mode, _extra in {
    "north_most": ("o tan cung phia bac", "o cao nhat", "nam phia tren cung", "o mep tren cung", "o dau ban do",
                   "nam tren dinh", "o hang dau", "cao hon tat ca"),
    "south_most": ("o tan cung phia nam", "o thap nhat", "nam phia duoi cung", "o mep duoi cung", "o cuoi ban do",
                   "nam duoi day", "o hang cuoi", "thap hon tat ca"),
    "west_most": ("o tan cung phia tay", "o xa ve ben trai nhat", "nam phia trai cung", "o mep trai cung",
                  "o cot dau", "nam sat trai", "o le trai", "lech ve trai nhat"),
    "east_most": ("o tan cung phia dong", "o xa ve ben phai nhat", "nam phia phai cung", "o mep phai cung",
                  "o cot cuoi", "nam sat phai", "o le phai", "lech ve phai nhat"),
}.items():
    MOST_PHRASES[_mode] += _extra
MOST_FRAMES += ("{G} nam {M}", "{G} {M} cua campus", "{G} {M} trong khuon vien")
VIA_FRAMES_BEFORE += (
    "truoc het phai qua {W}, xong",
    "dung chan o {W} lay {I2} roi",
    "di vong qua {W} truoc, roi",
    "buoc dau ghe {W}, sau do",
    "ban vao {W} lay {I2} da, roi",
)
VIA_FRAMES_AFTER += (
    "tren duong tien ghe {W} nhe",
    "nho qua {W} truoc",
    "truoc do ghe {W} lay {I2}",
    "phai qua {W} lay {I2} da",
    "nho ghe ngang {W}",
)
DISTRACTOR_FRAMES += (
    "{X} khong phai diem giao",
    "dung giao nham sang {X}",
    "khong can di ngang {X}",
    "{X} dang sua chua, tranh ra",
    "nguoi nhan khong con o {X}",
    "tuan truoc da giao {X} roi",
)
REDIRECT_PREFIX += (
    "thay doi: khong giao {X} nua,",
    "cap nhat: bo {X},",
    "dinh chinh: khong phai {X} ma",
    "quen don cu o {X} di,",
)
URGENT_TRUE += ("can trong 5 phut.", "dang gap!", "rat khan.", "nhanh nhat co the.", "ngay bay gio nhe.")
URGENT_FALSE += ("khong voi dau.", "thong tha.", "tu tu thoi.", "khi nao tien thi giao.")
FRAGILE_TRUE += ("hang de vo lam.", "ben trong la chen su.", "do de nut vo.", "nhe tay giup.", "co do thuy tinh.")
FRAGILE_FALSE += ("hang khong de vo dau.", "do chac lam.", "khong so vo.")
GREETINGS += ("chao!", "robot giao hang oi,", "nho robot:", "thong bao don:")
CLOSINGS += ("cam on robot.", "thank you.", "nho ky nhe.")


# ---------------------------------------------------------------- v3 additions

SPATIAL_PERSON_FRAMES = (
    "{P} o {L} dang cho {I}",
    "{P} o {L} can {I}",
    "{V} {I} cho {P} o {L}",
    "{P} dang doi {I} tai {L}",
    "nguoi nhan la {P}, dang o {L}",
    "{P} hien dang o {L}, nho {V} {I} {T} do",
    "{P} vua chuyen sang {L}, giao {I} cho ho",
)


def _phrase_index() -> tuple[re.Pattern, re.Pattern, dict[str, tuple[str, str]]]:
    """Replaceable phrases (place aliases, extreme-position phrases) and phrases that block them."""
    table: dict[str, tuple[str, str]] = {}
    for kind in TYPES:
        for phrase in set(lx.PLACE_ALIASES[kind]) | set(EXTRA_PLACES[kind]):
            table.setdefault(phrase, ("place", kind))
    for mode, phrases in MOST_PHRASES.items():
        for phrase in set(phrases) | set(lx.EXTREME_PHRASES[mode]):
            table.setdefault(phrase, ("most", mode))
    blockers = set(lx.ITEMS) | {p for kind in TYPES for p in set(lx.PERSON_ALIASES[kind]) | set(EXTRA_PERSONS[kind])}

    def alternation(phrases):
        body = "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))
        return re.compile(rf"(?<![a-z0-9])(?:{body})(?![a-z0-9])")

    return alternation(table), alternation(blockers), table


_INDEX = None


def augment_real(text: str, rng: random.Random, rate: float = 0.7) -> str:
    """Re-phrase a real annotated mission without changing its labels.

    Place names become another name of the same type and extreme-position phrases
    another phrase of the same direction (train phrase banks only), inside the real
    sentence structure. Names inside item or person phrases are left alone.
    """
    global _INDEX
    if _INDEX is None:
        _INDEX = _phrase_index()
    pattern, blockers, table = _INDEX
    folded = fold(text)
    blocked = [m.span() for m in blockers.finditer(folded)]
    out, last = [], 0
    for match in pattern.finditer(folded):
        start, end = match.span()
        if any(s < end and start < e for s, e in blocked) or rng.random() > rate:
            continue
        group, key = table[match.group(0)]
        bank = _places(key, False) if group == "place" else _split(MOST_PHRASES[key], False)
        out.append(folded[last:start])
        out.append(rng.choice(bank))
        last = end
    out.append(folded[last:])
    return "".join(out)


# ---------------------------------------------------------------- holdout split

def _split(bank: tuple[str, ...], holdout: bool) -> tuple[str, ...]:
    part = tuple(phrase for index, phrase in enumerate(bank) if (index % 5 == 4) == holdout)
    return part or bank


def _places(kind: str, holdout: bool) -> tuple[str, ...]:
    return _split(tuple(lx.PLACE_ALIASES[kind]) + EXTRA_PLACES[kind], holdout)


def _persons(kind: str, holdout: bool) -> tuple[str, ...]:
    return _split(tuple(lx.PERSON_ALIASES[kind]) + EXTRA_PERSONS[kind], holdout)


# ---------------------------------------------------------------- generation

class Generator:
    def __init__(self, seed: int = 0, holdout: bool = False) -> None:
        self.rng = random.Random(seed)
        self.holdout = holdout

    def pick(self, bank: tuple[str, ...]) -> str:
        return self.rng.choice(_split(bank, self.holdout))

    def place(self, kind: str) -> str:
        return self.rng.choice(_places(kind, self.holdout))

    def item(self) -> str:
        return self.rng.choice(tuple(lx.ITEMS) + EXTRA_ITEMS)

    def other_type(self, *exclude: str | None) -> str:
        return self.rng.choice([kind for kind in TYPES if kind not in exclude])

    def _goal_mode(self) -> str:
        weights = {
            "named": 30, "anchor_near": 30, "north_most": 5, "south_most": 5, "west_most": 5, "east_most": 5,
            "near": 5, "far": 4, "north": 2, "south": 2, "west": 2, "east": 2,
        }
        return self.rng.choices(list(weights), weights=list(weights.values()))[0]

    def _via_mode(self) -> str:
        weights = {"none": 50, "named": 40, "near": 3, "far": 3, "north": 1, "south": 1, "west": 1, "east": 1}
        return self.rng.choices(list(weights), weights=list(weights.values()))[0]

    def location(self, mode: str, kind: str | None, anchor: str | None) -> str:
        """Phrase describing a target for one mode."""
        r = self.rng
        if mode == "named":
            return self.place(kind)
        if mode in DIRECTIONS:
            words = _split(DIR_WORDS[mode], self.holdout)
            word = r.choice(words)
            return self.pick(DIR_FRAMES).format(P=self.place(kind), D=word, d=word.split()[-1])
        if mode == "near":
            return self.pick(NEAR_FRAMES).format(P=self.place(kind), A=self.place(anchor))
        if mode == "far":
            return self.pick(FAR_FRAMES).format(P=self.place(kind), A=self.place(anchor))
        if mode == "anchor_near":
            return self.pick(ANCHOR_NEAR_FRAMES).format(G=self.pick(GENERIC_HEADS), A=self.place(anchor))
        return self.pick(MOST_FRAMES).format(G=self.pick(GENERIC_HEADS), M=self.pick(MOST_PHRASES[mode])).strip()

    def example(self) -> Example:
        r = self.rng
        mode = self._goal_mode()
        map_only = mode == "anchor_near" or mode.endswith("_most")
        kind = None if map_only else r.choice(TYPES)
        anchor = self.other_type(kind) if mode in ("near", "far", "anchor_near") else None
        goal = TargetSpec(kind, None if mode == "named" else mode, anchor)

        # The goal sentence; named goals are sometimes given by a person.
        location = self.location(mode, kind, anchor)
        tail = r.choice(POLITE_TAILS)
        fill = dict(V=self.pick(DELIVER_VERBS), I=self.item(), T=self.pick(TO_WORDS), L=location, tail=tail)
        if mode == "named" and r.random() < 0.25:
            person = r.choice(_persons(kind, self.holdout))
            frame = self.pick(PERSON_FRAMES)
            fill["L"] = self.place(kind)
            core = frame.format(P=person, **fill)
        elif r.random() < 0.25:
            # A person at a described location: the location decides, the person's workplace does not.
            person = r.choice(_persons(r.choice(TYPES), self.holdout))
            core = self.pick(SPATIAL_PERSON_FRAMES).format(P=person, **fill)
        else:
            core = self.pick(GOAL_FRAMES).format(**fill)
        core = " ".join(core.split())

        # Redirect: "cancel X, instead: <core>" names a wrong type first.
        used = {kind, anchor}
        if r.random() < 0.3:
            wrong = self.other_type(*used)
            used.add(wrong)
            core = self.pick(REDIRECT_PREFIX).format(X=self.place(wrong)) + " " + core

        # Via.
        via_mode = self._via_mode()
        via = None
        sentences_before: list[str] = []
        sentences_after: list[str] = []
        if via_mode != "none":
            via_kind = self.other_type(*used)
            via_anchor = self.other_type(via_kind, *used) if via_mode in ("near", "far") else None
            used.update((via_kind, via_anchor))
            via = TargetSpec(via_kind, None if via_mode == "named" else via_mode, via_anchor)
            where = self.location(via_mode, via_kind, via_anchor)
            fill_via = dict(W=where, I2=self.item())
            if r.random() < 0.5:
                core = self.pick(VIA_FRAMES_BEFORE).format(**fill_via) + " " + core
            else:
                sentences_after.append(self.pick(VIA_FRAMES_AFTER).format(**fill_via) + ".")

        # Distractors naming other types.
        for _ in range(r.choice((0, 1, 1, 2, 2, 3))):
            sentences_before.append(self.pick(DISTRACTOR_FRAMES).format(X=self.place(self.other_type(*used))) + ".")

        urgent_choice = r.random()
        urgent = urgent_choice < 0.45
        if urgent:
            sentences_after.append(self.pick(URGENT_TRUE))
        elif urgent_choice < 0.75:
            sentences_after.append(self.pick(URGENT_FALSE))
        fragile_choice = r.random()
        fragile = fragile_choice < 0.4
        if fragile:
            sentences_after.append(self.pick(FRAGILE_TRUE))
        elif fragile_choice < 0.7:
            sentences_after.append(self.pick(FRAGILE_FALSE))

        sentences = sentences_before + [core.rstrip(",") + "."] + sentences_after
        if r.random() < 0.5:
            r.shuffle(sentences)
        greeting = r.choice(GREETINGS).format(n=r.randint(1000, 9999))
        closing = r.choice(CLOSINGS)
        text = " ".join(part for part in [greeting, *sentences, closing] if part)
        return Example(self.noise(text), goal, via, urgent, fragile)

    # ------------------------------------------------------------ noise

    def noise(self, text: str) -> str:
        r = self.rng
        if r.random() < 0.3:
            text = "".join(ch for ch in text if ch not in ".!?;()[]:,")
        rate = r.choice((0.0, 0.0, 0.03, 0.06, 0.1))
        if rate:
            words = text.split(" ")
            for index, word in enumerate(words):
                if len(word) > 2 and r.random() < rate:
                    words[index] = _typo(word, r)
            text = " ".join(words)
        case = r.random()
        if case < 0.4:
            text = ". ".join(part[:1].upper() + part[1:] for part in text.split(". "))
        elif case < 0.5:
            text = text.title()
        elif case < 0.55:
            text = text.upper()
        return " ".join(text.split())


def _typo(word: str, r: random.Random) -> str:
    i = r.randrange(len(word) - 1)
    if r.random() < 0.5:
        return word[:i] + word[i + 1 :]
    return word[:i] + word[i + 1] + word[i] + word[i + 2 :]


def generate(count: int, seed: int = 0, holdout: bool = False) -> list[Example]:
    generator = Generator(seed, holdout)
    return [generator.example() for _ in range(count)]


def spec_from_mission(mission) -> tuple[TargetSpec, TargetSpec | None]:
    """Parser-level labels from a scenes.json mission annotation."""

    def one(kind, ref):
        if kind is None:
            return None
        if ref is None:
            return TargetSpec(kind)
        if ref.kind == "anchor_near" or ref.kind.endswith("_most"):
            return TargetSpec(None, ref.kind, ref.anchor)
        return TargetSpec(kind, ref.kind, ref.anchor)

    return one(mission.goal, mission.goal_ref), one(mission.via, mission.via_ref)


# ---------------------------------------------------------------- expansion (v4)
# AGY supplied raw phrase candidates only. They were assembled locally and
# validated by scripts/nlp/v4_data_pipeline.py before this append-only block.

V4_BASE_SIZES = {
    "VIA_FRAMES_BEFORE": len(VIA_FRAMES_BEFORE),
    "VIA_FRAMES_AFTER": len(VIA_FRAMES_AFTER),
    "ANCHOR_NEAR_FRAMES": len(ANCHOR_NEAR_FRAMES),
    "GOAL_FRAMES": len(GOAL_FRAMES),
    "DISTRACTOR_FRAMES": len(DISTRACTOR_FRAMES),
    "REDIRECT_PREFIX": len(REDIRECT_PREFIX),
    "MOST_PHRASES": {key: len(value) for key, value in MOST_PHRASES.items()},
    "EXTRA_PLACES": {key: len(value) for key, value in EXTRA_PLACES.items()},
    "EXTRA_PERSONS": {key: len(value) for key, value in EXTRA_PERSONS.items()},
}

VIA_FRAMES_BEFORE += (
    "di den {W} lay {I2} da roi",
    "ghe vao {W} nhat {I2} roi moi",
    "re sang {W} xong hay",
    "vong qua {W} kiem {I2} roi",
    "tat vao {W} lay {I2} roi moi",
    "dung chan o {W} lay {I2} xong moi",
)

VIA_FRAMES_AFTER += (
    "nhung phai qua {W} lay {I2} truoc",
    "dong thoi tat qua {W} lay {I2} da",
    "va can re vao {W} lay {I2} roi moi tiep tuc",
    "voi dieu kien qua {W} lay {I2} truoc da",
    "tuy nhien hay vong qua {W} lay {I2} truoc",
    "dong thoi ghe vao {W} lay {I2} da roi hay giao",
)

ANCHOR_NEAR_FRAMES += (
    "xac dinh {G} ap sat moc {A} nhat",
    "khoanh vung {G} cach {A} doan ngan nhat",
    "truy tim {G} ke can moc {A} hon ca",
    "lay {G} voi khoang cach toi thieu ve phia {A}",
    "danh dau {G} tiep giap gan {A} hon tat ca",
    "xet {G} co do dai duong di toi {A} ngan nhat",
    "rut ra {G} cach diem tua {A} it buoc nhat",
    "phan loai {G} tiep can gan nhat toi moc {A}",
    "chon {G} co lo trinh toi {A} ngan hon tat ca",
    "hay tim {G} cach {A} it buoc di nhat",
    "{G} co cu ly toi {A} nho nhat",
    "trong cac lua chon lay {G} gan {A} hon ca",
)

GOAL_FRAMES += (
    "chu y {V} ho {I} toi {L} {tail}",
    "lam phien mang {I} sang tan {L} {tail}",
    "can gap {I} truoc cua {L} {tail}",
    "chuyen huong {V} sang {L} ngay {tail}",
    "tha {I} xuong {L} gium {tail}",
    "dung dung cho cu nua chay sang {L} {tail}",
    "khach hen lay {I} o {L} {tail}",
    "cam phien {V} {I} den thang {L} {tail}",
    "cuoi cung van phai {V} ve {L} {tail}",
    "giao nham roi phai mang vao {L} moi dung {tail}",
    "tinh hinh thay doi nen {V} sang {L} {tail}",
    "mang ho {I} re vao {L} {tail}",
    "chay thang toi {L} roi {V} {I} {tail}",
    "tot nhat la mang thang {I} vao {L} {tail}",
    "huong di moi la sang {L} de {V} {I} {tail}",
)

DISTRACTOR_FRAMES += (
    "chuyen nay chi di ngang {X}, khong ghe vao",
    "minh chi luot qua khu vuc {X}, khong dung chan",
    "xe chi chay ngang tam {X}, do khong phai cho muon toi",
    "doan duong chay qua {X}, khong co y dinh dung lai",
    "lo trinh co di qua {X} nhung khong dung chan o do",
    "cho {X} chi la moc de re, khong phai noi can tim",
    "chuyen nay xuat phat tu {X} di tiep, khong phai diem cuoi",
    "huong nay chi chay song song voi {X}, khong re vao do",
)

REDIRECT_PREFIX += (
    "toi da tu bo y dinh den {X}, ma se",
    "thoi bo y nghi qua {X} di, gio",
    "thay doi lo trinh da chon o {X}, gio hay",
    "quen diem {X} toi vua doc luc nay di, gio hay",
)

for _mode, _extra in {
    "north_most": (
        "chon toa do lech han ve phia bac",
        "danh dau diem moc tien sau nhat ve phuong bac",
        "huong thang ve phia bac xa hon bat ky vi tri nao",
        "tim toa do tien gan cuc bac dia ly cua ban do",
        "chon o dat lech cao nhat ve huong bac",
        "khu vuc tien phong ve phia bac so voi phan con lai",
        "cham moc nhin thang ve huong chinh bac xa nhat",
        "toa do lech sau nhat sang phuong bac",
        "chi toi o dat vuot qua moi moc khac ve phia bac",
        "diem moc nhin ra bien gioi phia bac xa hon tat ca",
    ),
    "south_most": (
        "di chuyen xuong phia nam sau hon tat ca",
        "tim toa do lech han ve phuong nam",
        "xac dinh khu vuc do sau xuong phia nam cua ban do",
        "danh dau moc cham day phuong nam xa nhat",
        "huong ve phuong nam voi khoang lech lon hon moi diem",
        "chon toa do tien sat ranh gioi phia nam",
        "dinh vi o lech xuong phia nam xa nhat he thong",
        "truy tim diem co vi do thap nhat tren ban do",
        "chon khu vuc vuot qua moi diem khac ve phia nam",
        "chi ra vi tri nga ve phuong nam xa nhat",
    ),
    "west_most": (
        "tien sat ve phuong tay xa hon toan bo",
        "chon diem lech han ve phia mat troi lan",
        "xac dinh toa do lech sau nhat sang phia tay",
        "danh dau moc do ve phuong tay sau hon ca",
        "huong thang ve phia tay xa hon moi vi tri khac",
        "tim diem co kinh do nho nhat tren ban do",
        "lay toa do vuon sau nhat ve huong tay",
        "truy tim o dat nga ve phuong tay sau nhat",
        "chon khu vuc lech sang tay vuot qua tat ca",
        "toa do can ve huong tay nhieu nhat toan vung",
    ),
    "east_most": (
        "tien sat ve phuong dong vuot troi hon ca",
        "chon diem lech han ve phia mat troi moc",
        "xac dinh toa do lech sau nhat sang phia dong",
        "danh dau moc do ve phuong dong sau hon tat ca",
        "huong thang ve phuong dong xa hon moi vi tri",
        "tim diem co kinh do lon nhat tren ban do",
        "lay toa do vuon xa nhat sang huong dong",
        "truy tim o dat nga ve phia dong sau nhat",
        "chon khu vuc lech sang dong vuot qua toan bo",
        "toa do vuon sang huong dong nhieu nhat toan vung",
    ),
}.items():
    MOST_PHRASES[_mode] += _extra

for _kind, _extra in {
    "library": (
        "khu vuc doc sach va tra cuu tai lieu",
        "noi muon tra sach giao trinh",
        "khong gian doc yen tinh tang tren",
        "quay tra cuu luan van do an",
    ),
    "dorm": (
        "khu noi tru sinh vien tap trung",
        "day nha o sinh vien noi tru",
        "toa nha luu tru sinh vien",
        "khu nha o tap the sinh vien",
    ),
    "sports": (
        "khu luyen tap the duc the thao",
        "san van dong ngoai troi",
        "nha thi dau the thao da nang",
        "san tap bong ro va cau long",
    ),
    "clinic": (
        "phong cham soc suc khoe ban dau",
        "tram y te cham soc sinh vien",
        "khu vuc cap cuu va so cuu hoc duong",
        "quay phat thuoc va kham ban dau",
    ),
    "canteen": (
        "khu vuc phuc vu an uong sinh vien",
        "nha an tap the khuon vien truong",
        "quay ban com trua va do an nhanh",
        "khu phuc vu bua trua sinh vien",
    ),
    "parking": (
        "khu vuc trong giu xe hai banh",
        "bai gui xe may sinh vien",
        "tang ham de xe khu giang duong",
        "bai trong giu phuong tien tap trung",
    ),
    "lecture": (
        "toa nha phong hoc ly thuyet",
        "day phong hoc giang duong lon",
        "hoi truong hoc tap trung sinh vien",
        "khu vuc lop hoc ly thuyet",
    ),
    "lab": (
        "phong thuc hanh may tinh va thi nghiem",
        "khu nghien cuu va thi nghiem chuyen sau",
        "xuong thuc hanh ky thuat che tao",
        "phong mo phong thi nghiem thuc hanh",
    ),
    "office": (
        "toa nha hanh chinh quan ly truong",
        "phong tiep don va giai quyet thu tuc sinh vien",
        "khu vuc van phong cac khoa chuyen mon",
        "bo phan mot cua giai quyet ho so",
    ),
    "gate": (
        "cong chinh ra vao khuon vien truong",
        "cong phu phia sau giang duong",
        "loi vao danh cho phuong tien co gioi",
        "diem kiem soat the ra vao khuon vien",
    ),
}.items():
    EXTRA_PLACES[_kind] += _extra

for _kind, _extra in {
    "library": ("can bo thu vien phu trach kho sach", "thu thu huong dan muon tra sach"),
    "dorm": ("ban quan ly khu noi tru sinh vien", "bac bao ve truc toa nha noi tru"),
    "sports": ("giang vien bo mon giao duc the chat", "huan luyen vien doi tuyen the thao truong"),
    "clinic": ("bac si truc tram y te truong", "can bo y te phu trach suc khoe sinh vien"),
    "canteen": ("nhan vien phuc vu nha an truong", "dau bep nau com trua sinh vien"),
    "parking": ("bac bao ve trong giu xe truong", "nhan vien soat ve bai giu xe"),
    "lecture": ("giang vien dung lop giang duong", "tro giang phu trach gio ly thuyet"),
    "lab": ("can bo phu trach phong thuc hanh", "ky thuat vien phong thi nghiem"),
    "office": ("chuyen vien phong quan ly dao tao", "can bo tiep nhan ho so sinh vien"),
    "gate": ("bao ve truc cong ra vao truong", "nhan vien an ninh truc cong chinh"),
}.items():
    EXTRA_PERSONS[_kind] += _extra
