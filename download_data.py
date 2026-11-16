import os, csv, shutil, random
import kagglehub

CLASS_MAP = {
    14: "dur", 13: "yol_ver", 17: "girilmez",
    2: "hiz_siniri_50", 27: "yaya_dikkat", 34: "sola_mecburi",
}
MAX_TRAIN = 200   
MAX_TEST = 60     
EXTRA_DIR = "extra"
IMG_EXT = (".png", ".jpg", ".jpeg", ".ppm", ".webp")
random.seed(42)

if os.path.isdir("dataset"):
    shutil.rmtree("dataset")

path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
train_dir, test_csv = None, None
for root, dirs, files in os.walk(path):
    if train_dir is None and "14" in dirs and "0" in dirs:
        train_dir = root
    if test_csv is None and "Test.csv" in files:
        test_csv = os.path.join(root, "Test.csv")
assert train_dir and test_csv, "GTSRB yapısı bulunamadı"
base = os.path.dirname(test_csv)

for cid, name in CLASS_MAP.items():
    src = os.path.join(train_dir, str(cid))
    dst = os.path.join("dataset", "train", name)
    os.makedirs(dst, exist_ok=True)
    imgs = [f for f in os.listdir(src) if f.lower().endswith(IMG_EXT)]
    random.shuffle(imgs)
    for f in imgs[:MAX_TRAIN]:
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))

rows = []
with open(test_csv, encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        if int(row["ClassId"]) in CLASS_MAP:
            rows.append(row)
random.shuffle(rows)
count = {name: 0 for name in CLASS_MAP.values()}
for row in rows:
    name = CLASS_MAP[int(row["ClassId"])]
    if count[name] >= MAX_TEST:
        continue
    dst = os.path.join("dataset", "test", name)
    os.makedirs(dst, exist_ok=True)
    shutil.copy(os.path.join(base, row["Path"]), dst)
    count[name] += 1

if os.path.isdir(EXTRA_DIR):
    for cls in sorted(os.listdir(EXTRA_DIR)):
        src = os.path.join(EXTRA_DIR, cls)
        if not os.path.isdir(src):
            continue
        imgs = [f for f in os.listdir(src) if f.lower().endswith(IMG_EXT)]
        random.shuffle(imgs)
        n_test = max(1, int(0.2 * len(imgs)))
        parts = {"test": imgs[:n_test], "train": imgs[n_test:]}
        for split, lst in parts.items():
            dst = os.path.join("dataset", split, cls)
            os.makedirs(dst, exist_ok=True)
            for f in lst:
                shutil.copy(os.path.join(src, f), os.path.join(dst, f))
else:
    print("UYARI: 'extra' klasörü yok")

for split in ("train", "test"):
    for cls in sorted(os.listdir(os.path.join("dataset", split))):
        print(split, cls, len(os.listdir(os.path.join("dataset", split, cls))))
print("Bitti.")