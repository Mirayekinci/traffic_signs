import os, hashlib, requests
from io import BytesIO
from PIL import Image
from ddgs import DDGS

queries = {
    "sola_donulmez": ["sola dönülmez trafik levhası", "no left turn traffic sign road"],
    "yaya_gecidi": ["yaya geçidi trafik levhası", "Turkey pedestrian crossing sign blue"],
    "trafik_lambasi": ["trafik lambası uyarı levhası", "traffic lights ahead warning sign"],
}
TARGET = 60
HEADERS = {"User-Agent": "Mozilla/5.0"}

for cls, qs in queries.items():
    out = os.path.join("extra", cls)
    os.makedirs(out, exist_ok=True)
    seen, n = set(), len(os.listdir(out))
    for q in qs:
        try:
            results = DDGS().images(q, max_results=100)
        except Exception as e:
            print("Arama hatası:", q, e)
            continue
        for r in results:
            if n >= TARGET:
                break
            try:
                resp = requests.get(r["image"], headers=HEADERS, timeout=8)
                h = hashlib.md5(resp.content).hexdigest()
                if h in seen:
                    continue
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                if min(img.size) < 100:
                    continue
                seen.add(h)
                img.save(os.path.join(out, f"{cls}_{n:03d}.jpg"))
                n += 1
            except Exception:
                continue
    print(cls, "->", n, "görsel")