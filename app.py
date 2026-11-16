import torch, torch.nn as nn
from torchvision import models, transforms
import gradio as gr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load("model/resnet_traffic.pth", map_location=device)
def new_func(ckpt):
    class_names = ckpt["classes"]
    return class_names

class_names = new_func(ckpt)
NUM_CLASSES = len(class_names)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(ckpt["state_dict"])
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict_traffic_sign(image):
    if image is None:
        return {}
    input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(input_tensor)[0], dim=0)
    return {class_names[i]: float(probs[i]) for i in range(NUM_CLASSES)}

demo = gr.Interface(
    fn=predict_traffic_sign,
    inputs=gr.Image(type="pil", label="Trafik İşareti Fotoğrafı Yükle"),
    outputs=gr.Label(num_top_classes=3, label="Model Tahmin Sonuçları"),
    title="Trafik İşareti Sınıflandırma Sistemi",
    description="Kırpılmış, tek levhalı fotoğraf yükleyin.",
)

if __name__ == "__main__":
    demo.launch()