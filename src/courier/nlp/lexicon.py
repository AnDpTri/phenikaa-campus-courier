"""Hand-written Vietnamese lexicon, accent-folded.

Phrases are written without diacritics because `text.fold` strips them from the
input. Everything here is general campus vocabulary or phrasing seen in train;
nothing is taken from test.
"""

from __future__ import annotations

import re
from functools import cache

# Names a place outright. Longest match wins, so "thu vien truong" beats "thu vien".
PLACE_ALIASES: dict[str, tuple[str, ...]] = {
    "library": (
        "thu vien", "thu vien truong", "khu doc sach", "phong doc", "phong doc sach",
        "cho tra sach", "noi tra sach", "noi muon giao trinh", "noi muon sach", "noi muon sach tham khao",
        "cho muon sach", "phong tai lieu", "trung tam thong tin thu vien",
    ),
    "dorm": (
        "ky tuc xa", "ktx", "khu ktx", "khu noi tru", "phong o sinh vien",
        "toa nha o cua sinh vien", "khu o cua sinh vien", "nha o sinh vien",
        "khu nha o sinh vien", "day phong noi tru", "khu ky tuc",
    ),
    "sports": (
        "nha the chat", "nha thi dau", "san bong", "san tap", "san the thao",
        "khu the thao", "san van dong", "phong gym", "nha tap", "khu van dong",
        "san luyen tap", "trung tam the thao",
    ),
    "clinic": (
        "tram y te", "phong y te", "khu y te", "tram xa", "phong so cuu",
        "phong kham", "phong kham cua truong", "phong bac si", "trung tam y te",
        "noi kham benh", "benh xa",
    ),
    "canteen": (
        "can tin", "cang tin", "canteen", "nha an", "bep an", "khu an uong",
        "phong an", "nha an tap the", "phong an tap the", "khu am thuc", "bep truong hoc",
    ),
    "parking": (
        "bai xe", "bai do xe", "bai gui xe", "nha xe", "ham xe", "cho de xe",
        "khu gui xe", "cho gui xe", "noi gui xe", "cho do xe", "noi do xe", "khu do xe",
        "khu dau xe", "noi giu xe", "nha de xe",
    ),
    "lecture": (
        "giang duong", "khu giang duong", "toa giang duong", "hoi truong", "hoi truong hoc",
        "lop hoc", "phong hoc", "phong hoc lon", "khu lop hoc", "day phong hoc", "toa nha hoc",
    ),
    "lab": (
        "phong thi nghiem", "khu thi nghiem", "phong lab", "lab", "lab hoa",
        "phong thuc hanh", "xuong thuc hanh", "phong thuc nghiem", "phong nghien cuu",
        "khu thuc hanh",
    ),
    "office": (
        "phong hanh chinh", "khu hanh chinh", "phong dao tao", "phong mot cua",
        "toa hieu bo", "van phong khoa", "van phong", "phong giao vu", "toa hanh chinh",
        "nha hieu bo", "ban giam hieu",
    ),
    "gate": (
        "cong truong", "cong chinh", "cong a", "cong b", "cong vao", "loi vao truong",
        "loi vao", "chot bao ve cong", "chot bao ve", "chot cong", "cong phu", "cong sau",
    ),
}

# Function words that pick out a type when a place is described by what happens there,
# e.g. "noi kham suc khoe". Used only after a generic head noun (see HEAD_NOUNS).
PLACE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "library": ("sach", "giao trinh", "doc", "thu vien", "tai lieu tham khao", "tra cuu", "muon sach"),
    "dorm": ("noi tru", "ky tuc", "ktx", "o cua sinh vien", "sinh vien o", "ngu", "o sinh vien"),
    "sports": ("the thao", "the chat", "thi dau", "bong", "van dong", "luyen tap", "tap luyen", "gym", "tap the duc"),
    "clinic": ("y te", "kham", "bac si", "so cuu", "suc khoe", "benh", "thuoc", "y ta"),
    "canteen": ("an uong", "bua trua", "bua sang", "bua toi", "bua an", "nau an", "an com", "com", "an", "phuc vu bua"),
    "parking": ("xe", "gui xe", "do xe", "dau xe", "giu xe"),
    "lecture": ("len lop", "giang", "lop", "hoc", "nghe giang", "giang day"),
    "lab": ("thi nghiem", "thuc hanh", "thuc nghiem", "lab", "nghien cuu", "hoa chat"),
    "office": ("hanh chinh", "dao tao", "mot cua", "giao vu", "hieu bo", "ho so", "giay to", "cong van", "thu tuc"),
    "gate": ("cong", "bao ve", "khach den", "loi vao", "ra vao", "don khach"),
}

HEAD_NOUNS = ("noi", "cho", "khu", "phong", "toa", "toa nha", "nha", "day", "san", "tram", "khu vuc", "diem")

# People whose workplace implies the type. Weaker than a named place in the same clause.
PERSON_ALIASES: dict[str, tuple[str, ...]] = {
    "library": ("thu thu", "can bo thu vien", "nhan vien thu vien"),
    "dorm": ("ban quan ly ky tuc xa", "sinh vien noi tru", "quan ly ktx", "quan ly ky tuc xa"),
    "sports": ("huan luyen vien", "doi bong", "giao vien the duc", "giang vien the chat"),
    "clinic": ("nhan vien y te", "bac si truong", "bac si", "y ta"),
    "canteen": ("bep truong", "co cap duong", "dau bep", "co nau an"),
    "parking": ("bac giu xe", "nhan vien trong xe", "chu giu xe"),
    "lecture": ("giang vien dang day", "giang vien", "lop truong", "thay giao", "co giao"),
    "lab": ("ky thuat vien phong lab", "ky thuat vien", "nhom nghien cuu"),
    "office": ("thu ky khoa", "can bo phong dao tao", "can bo hanh chinh", "chuyen vien"),
    "gate": ("to don khach", "bao ve cong", "bac bao ve", "bao ve"),
}

# Goods that contain place words; masked so "the thu vien" is not read as the library.
ITEMS = (
    "the thu vien", "tui so cuu", "the giu xe", "bo dung cu thi nghiem", "dung cu thi nghiem",
    "sach tham khao", "chia khoa xe", "bom xe", "mu bao hiem", "chong giao trinh",
    "hop sach tra", "dung cu tap luyen", "tap tai lieu", "tui do giat", "khay com",
    "hop thuoc", "bo bang gac", "ho so", "phong bi cong van", "tap giay to", "nguyen lieu nau an",
    "thung thuc pham", "thung nuoc uong", "ket nuoc ngot", "linh kien dien tu", "hoa chat",
    "qua bong", "bo vot", "may chieu", "micro khong day", "hop phan", "tap bai kiem tra",
    "bien ten khach", "so ghi khach", "don hang can tra", "hop khau trang", "con dau", "mau vat",
    "bo dam", "ve xe", "sach", "giao trinh",
)

# Map-only goal descriptions, "the place furthest north", etc.
EXTREME_PHRASES: dict[str, tuple[str, ...]] = {
    "north_most": (
        "ria tren nhat", "o ria tren", "ria tren", "mep tren nhat", "goc tren nhat",
        "cao nhat", "tren cung", "xa nhat ve phia bac", "xa nhat ve phia tren", "phia bac nhat",
        "bac nhat", "ngoai cung ben tren", "ngoai cung phia tren", "sat mep tren nhat",
        "sat mep tren", "tren nhat", "o tren cung",
    ),
    "south_most": (
        "ria duoi nhat", "o ria duoi", "ria duoi", "mep duoi nhat", "goc duoi nhat",
        "thap nhat", "duoi cung", "xa nhat ve phia nam", "xa nhat ve phia duoi", "phia nam nhat",
        "ngoai cung ben duoi", "ngoai cung phia duoi", "sat mep duoi nhat", "sat mep duoi",
        "duoi nhat", "o duoi cung",
    ),
    "west_most": (
        "ria trai nhat", "o ria trai", "ria trai", "mep trai nhat", "goc trai nhat",
        "ngoai cung ben trai", "ngoai cung phia trai", "sat mep trai nhat", "sat mep trai",
        "xa nhat ve phia tay", "xa nhat ve ben trai", "xa nhat ve phia trai", "phia tay nhat",
        "tay nhat", "trai nhat", "ben trai cung", "trai cung",
    ),
    "east_most": (
        "ria phai nhat", "o ria phai", "ria phai", "mep phai nhat", "goc phai nhat",
        "ngoai cung ben phai", "ngoai cung phia phai", "sat mep phai nhat", "sat mep phai",
        "xa nhat ve phia dong", "xa nhat ve ben phai", "xa nhat ve phia phai", "phia dong nhat",
        "dong nhat", "phai nhat", "ben phai cung", "phai cung",
    ),
}

DIRECTIONS = {
    "bac": "north", "tren": "north",
    "nam": "south", "duoi": "south",
    "tay": "west", "trai": "west",
    "dong": "east", "phai": "east",
}

# Statements about urgency and fragility. A sentence matching a FORCE phrase decides
# the flag outright; otherwise a WORD decides it, inverted when a negator precedes it.
URGENT_FORCE_TRUE = (
    "khong duoc cham tre", "khong duoc cham", "khong the cham", "khong duoc tre", "khong the doi",
    "khong cho duoc", "khong duoc de lau", "duoc cham tre",
)
URGENT_FORCE_FALSE = (
    "khong gap", "khong voi", "khong can voi", "tu tu", "cung duoc", "cung kip", "thong tha",
    "khong can nhanh", "khong khan cap", "khong hoa toc", "chua can", "khi nao ranh", "khong cap",
    "cham cung duoc", "tre cung duoc", "de sau", "khong quan trong thoi gian",
)
URGENT_WORDS = (
    "gap", "khan cap", "khan", "hoa toc", "cang nhanh cang tot", "nhanh", "can ngay", "di ngay",
    "giao ngay", "ngay lap tuc", "lap tuc", "ngay bay gio", "uu tien", "toc hanh", "som nhat",
    "cap bach", "cap toc", "khong tre", "dung tre", "som cang tot", "cang som", "trong vong",
)
FRAGILE_FORCE_FALSE = (
    "chang sao", "khong sao", "hang ben", "do ben", "khong lo vo", "khong so vo", "chac chan",
    "cung khong vo", "khong can nhe tay", "khong can can than", "tha thoai mai",
)
FRAGILE_WORDS = (
    "de vo", "vo", "de hong", "hong", "thuy tinh", "gom", "su", "can than", "nhe tay", "va dap",
    "va cham", "mong manh", "ky va dap", "nhe nhang", "fragile",
)
NEGATORS = frozenset(("khong", "chang", "cha", "chua", "dung", "ko", "k"))

# Clause cues. Patterns are regex fragments over accent-folded tokens joined by spaces;
# a mention is written as `@` in the explanations.
NEGATE_BEFORE = (
    r"khong (?:can )?(?:phai )?(?:ghe|qua|den|toi|giao|tat qua|di)(?: qua| vao| o| den| toi| sang)?",
    r"dung (?:nham|lan)(?: voi| sang| la| thanh)?",
    r"dung(?: co)?(?: di)?(?: qua| ghe| vao| toi| den| sang)(?: qua| vao| o)?",
    r"da roi(?: khoi)?",
    r"da giao(?: o| den| toi| cho| xong o)?",
    r"tin truoc(?: \w+){0,2} (?:ghi|noi|nhan|bao)(?: la)?",
    r"nham(?: la| thanh| sang| voi)?",
    r"huy(?: don)?(?: giao)?(?: o| den| toi)?",
    r"bo qua",
    r"khong (?:phai|pai|hai|phi|phia)(?: la)?(?: o)?",
    r"thay vi(?: giao)?(?: o| den| toi)?",
    r"khong con(?: o)?",
    r"khong nhan(?: o)?",
    r"tranh",
)
NEGATE_AFTER = (
    r"(?:la |bi )?nham",
    r"khong (?:phai|phia|pai|phi|hai) (?:la )?(?:diem|noi|cho|dia diem) (?:nhan|giao|den)",
    r"nua",
    r"khong phai o do",
    r"da xong",
    r"khong (?:can|dung)",
)
VIA_BEFORE = (
    r"ghe(?: qua| vao| ngang| tham| sang| lai)?(?: o)?",
    r"tat(?: qua| vao| ngang)",
    r"truoc (?:tien|het|da) (?:qua|den|toi|ghe|di|ra|vao|sang)(?: qua)?",
    r"di ngang(?: qua)?",
    r"ghe toi",
    r"di qua",
    r"(?:lay|nhan|lanh|lay ve)(?: \S+){0,4} (?:o|tai)",
)
# A mention followed by a pick-up and "truoc" is a stop-over even without "ghe".
VIA_AFTER = (
    r"(?:\S+ ){0,2}(?:lay|nhan|don|lay ve|nhan lai)(?: \S+){0,6} truoc\b",
    r"truoc(?: da| tien| het)?\b",
    r"(?:\S+ ){0,4}xong(?: thi| roi)?\b",
)
# Phrases that open a new clause. Missions normally separate clauses with
# punctuation; when it is missing, a cue from the next clause ("... bai xe Khong can
# ghe ...") would otherwise be read as part of the previous one. Splitting before
# these phrases is harmless when punctuation is present.
CLAUSE_STARTS = (
    "khong can ghe", "dung nham", "nguoi nhan", "hom qua", "huy don", "tin truoc", "luc nay",
    "thay vao do", "dung ra", "doi lai", "yeu cau moi", "xin chao", "chao robot", "robot oi",
    "nho ban", "nhan robot", "cam on", "thanks", "diem giao", "dich den", "chu y", "ben trong",
    "hang de vo", "hang chac chan", "hang ben", "hang ky", "do gom", "do khong", "nhe tay",
    "gap nhe", "hoa toc", "cang nhanh", "dang can gap", "can ngay", "viec nay", "khong gap", "khong can gap",
    "cu tu tu", "chieu nay", "mai giao", "uu tien", "khong duoc cham", "roi cung", "bo qua",
    "nhung phai", "truoc khi", "truoc tien", "kien hang", "khong can voi", "chua di thang",
)
# Grammar words the parser's regexes depend on; protected from typo repair.
CORE_WORDS = tuple(
    "man ria mep phia ben huong mien goc xong thi hon nhat gan xa sat canh ke cach nam o la nham dung khong "
    "phai can ghe giao toi den qua lay nhan truoc sau roi chu ma voi nhe nua tren duoi trai bac tay dong ban do "
    "dia diem noi cho khu vi tri hang".split()
)
# "the place <link> X nhat" = the landmark nearest to X.
NEAR_LINK = (
    r"(?:gan|sat|canh|ke|sat canh|ngay canh|ke ben|lien ke|sat ben|ben canh|nam canh|nam sat|nam gan|o gan|o canh)"
    r"(?: voi)?"
)
GENERIC_PLACE = r"(?:dia diem|noi|cho|toa nha|khu|diem|vi tri|toa|khu vuc)"


@cache
def lexicon_phrases() -> tuple[str, ...]:
    """Every literal multi-word phrase in the lexicon (regex cue patterns excluded)."""
    phrases: list[str] = []
    for table in (PLACE_ALIASES, PERSON_ALIASES, PLACE_KEYWORDS, EXTREME_PHRASES):
        for values in table.values():
            phrases.extend(values)
    for values in (
        ITEMS, HEAD_NOUNS, URGENT_FORCE_TRUE, URGENT_FORCE_FALSE, URGENT_WORDS, FRAGILE_FORCE_FALSE, FRAGILE_WORDS,
    ):
        phrases.extend(values)
    return tuple(phrases)


@cache
def lexicon_words() -> frozenset[str]:
    """Every word the lexicon relies on; the speller must never rewrite these into something else."""
    words = {word for phrase in lexicon_phrases() for word in phrase.split()}
    words.update(NEGATORS)
    words.update(CORE_WORDS)
    words.update(DIRECTIONS)
    for pattern in (*NEGATE_BEFORE, *NEGATE_AFTER, *VIA_BEFORE, *VIA_AFTER, NEAR_LINK, GENERIC_PLACE):
        words.update(re.findall(r"[a-z]{2,}", pattern))
    return frozenset(words)
