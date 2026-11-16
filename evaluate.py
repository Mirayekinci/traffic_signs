import os, time, torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

ckpt = torch.load("model/resnet_traffic.pth", map_location=DEVICE)
classes = ckpt["classes"]
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ckpt["state_dict"])
model.to(DEVICE).eval()

test_ds = datasets.ImageFolder("dataset/test", transform=tf)
dl = DataLoader(test_ds, batch_size=32)
y_true, y_pred = [], []
start = time.time()
with torch.no_grad():
    for x, y in dl:
        y_pred += model(x.to(DEVICE)).argmax(1).cpu().tolist()
        y_true += y.tolist()
total = time.time() - start

print("Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=test_ds.classes, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
print(f"Ortalama tahmin süresi: {1000*total/len(test_ds):.2f} ms/görüntü")

if os.path.isdir("test_images"):
    files = sorted(f for f in os.listdir("test_images")
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
else:
    files = []

if not files:
    print("test_images klasörü boş veya yok, telefon fotoğraflarını ekleyin.")
else:
    fig, axes = plt.subplots(1, len(files), figsize=(4 * len(files), 4.5))
    axes = [axes] if len(files) == 1 else list(axes)
    for ax, f in zip(axes, files):
        img = Image.open(os.path.join("test_images", f)).convert("RGB")
        t0 = time.time()
        with torch.no_grad():
            probs = torch.softmax(model(tf(img).unsqueeze(0).to(DEVICE)), 1)[0]
        ms = 1000 * (time.time() - t0)
        pred = classes[probs.argmax().item()]

        matches = [c for c in classes if f.lower().startswith(c)]
        true = max(matches, key=len) if matches else "sınıf dışı"

        ok = (pred == true)
        print(f"{f}: gerçek={true} | tahmin={pred} ({probs.max():.2f}) | {ms:.1f} ms | {'DOĞRU' if ok else 'YANLIŞ'}")
        ax.imshow(img); ax.axis("off")
        ax.set_title(f"Gerçek: {true}\nTahmin: {pred} ({probs.max():.2f})",
                     color="green" if ok else "red", fontsize=10)
    plt.tight_layout()
    plt.savefig("real_photo_results.png", dpi=150)
    plt.show()