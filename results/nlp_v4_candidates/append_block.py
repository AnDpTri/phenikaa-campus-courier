# NLP v4 expansion; generated from validated candidates.
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

MOST_PHRASES["north_most"] += (
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
)

MOST_PHRASES["south_most"] += (
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
)

MOST_PHRASES["west_most"] += (
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
)

MOST_PHRASES["east_most"] += (
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
)

EXTRA_PLACES["library"] += (
    "khu vuc doc sach va tra cuu tai lieu",
    "noi muon tra sach giao trinh",
    "khong gian doc yen tinh tang tren",
    "quay tra cuu luan van do an",
)

EXTRA_PLACES["dorm"] += (
    "khu noi tru sinh vien tap trung",
    "day nha o sinh vien noi tru",
    "toa nha luu tru sinh vien",
    "khu nha o tap the sinh vien",
)

EXTRA_PLACES["sports"] += (
    "khu luyen tap the duc the thao",
    "san van dong ngoai troi",
    "nha thi dau the thao da nang",
    "san tap bong ro va cau long",
)

EXTRA_PLACES["clinic"] += (
    "phong cham soc suc khoe ban dau",
    "tram y te cham soc sinh vien",
    "khu vuc cap cuu va so cuu hoc duong",
    "quay phat thuoc va kham ban dau",
)

EXTRA_PLACES["canteen"] += (
    "khu vuc phuc vu an uong sinh vien",
    "nha an tap the khuon vien truong",
    "quay ban com trua va do an nhanh",
    "khu phuc vu bua trua sinh vien",
)

EXTRA_PLACES["parking"] += (
    "khu vuc trong giu xe hai banh",
    "bai gui xe may sinh vien",
    "tang ham de xe khu giang duong",
    "bai trong giu phuong tien tap trung",
)

EXTRA_PLACES["lecture"] += (
    "toa nha phong hoc ly thuyet",
    "day phong hoc giang duong lon",
    "hoi truong hoc tap trung sinh vien",
    "khu vuc lop hoc ly thuyet",
)

EXTRA_PLACES["lab"] += (
    "phong thuc hanh may tinh va thi nghiem",
    "khu nghien cuu va thi nghiem chuyen sau",
    "xuong thuc hanh ky thuat che tao",
    "phong mo phong thi nghiem thuc hanh",
)

EXTRA_PLACES["office"] += (
    "toa nha hanh chinh quan ly truong",
    "phong tiep don va giai quyet thu tuc sinh vien",
    "khu vuc van phong cac khoa chuyen mon",
    "bo phan mot cua giai quyet ho so",
)

EXTRA_PLACES["gate"] += (
    "cong chinh ra vao khuon vien truong",
    "cong phu phia sau giang duong",
    "loi vao danh cho phuong tien co gioi",
    "diem kiem soat the ra vao khuon vien",
)

EXTRA_PERSONS["library"] += (
    "can bo thu vien phu trach kho sach",
    "thu thu huong dan muon tra sach",
)

EXTRA_PERSONS["dorm"] += (
    "ban quan ly khu noi tru sinh vien",
    "bac bao ve truc toa nha noi tru",
)

EXTRA_PERSONS["sports"] += (
    "giang vien bo mon giao duc the chat",
    "huan luyen vien doi tuyen the thao truong",
)

EXTRA_PERSONS["clinic"] += (
    "bac si truc tram y te truong",
    "can bo y te phu trach suc khoe sinh vien",
)

EXTRA_PERSONS["canteen"] += (
    "nhan vien phuc vu nha an truong",
    "dau bep nau com trua sinh vien",
)

EXTRA_PERSONS["parking"] += (
    "bac bao ve trong giu xe truong",
    "nhan vien soat ve bai giu xe",
)

EXTRA_PERSONS["lecture"] += (
    "giang vien dung lop giang duong",
    "tro giang phu trach gio ly thuyet",
)

EXTRA_PERSONS["lab"] += (
    "can bo phu trach phong thuc hanh",
    "ky thuat vien phong thi nghiem",
)

EXTRA_PERSONS["office"] += (
    "chuyen vien phong quan ly dao tao",
    "can bo tiep nhan ho so sinh vien",
)

EXTRA_PERSONS["gate"] += (
    "bao ve truc cong ra vao truong",
    "nhan vien an ninh truc cong chinh",
)
