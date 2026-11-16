import os, time, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH, EPOCHS, LR = 32, 15, 1e-4
torch.manual_seed(42)
os.makedirs("model", exist_ok=True)

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

full_train = datasets.ImageFolder("dataset/train", transform=train_tf)
full_val = datasets.ImageFolder("dataset/train", transform=eval_tf)
n = len(full_train)
idx = torch.randperm(n).tolist()
split = int(0.8 * n)
train_ds = Subset(full_train, idx[:split])
val_ds = Subset(full_val, idx[split:])

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH)

classes = full_train.classes
print("Sınıflar:", classes, "| train:", len(train_ds), "| val:", len(val_ds))

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

best_acc, start = 0.0, time.time()
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in train_dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
    val_acc = correct / len(val_ds)
    print(f"Epoch {epoch+1}/{EPOCHS} | loss {total_loss/len(train_ds):.4f} | val acc {val_acc:.4f}")

    if val_acc >= best_acc:
        best_acc = val_acc
        torch.save({"state_dict": model.state_dict(), "classes": classes},
                   "model/resnet_traffic.pth")

print(f"En iyi val acc: {best_acc:.4f} | Eğitim süresi: {time.time()-start:.1f} sn")