# Türk Trafik Levhası Sınıflandırma (ResNet18)

BIL485 Derin Öğrenme ödevi. ImageNet ön eğitimli ResNet18 ile 9 sınıflı trafik levhası sınıflandırması ve Gradio arayüzü.

## Sonuçlar

| Metrik | Değer |
|---|---|
| Test accuracy | 0.9740 |
| Macro F1 | 0.9289 |
| Weighted F1 | 0.9769 |
| Tahmin süresi | 90.46 ms/görüntü ([CPU/GPU]) |
| Eğitim süresi | 2140.3 sn (15 epoch) |
| Telefon fotoğrafları | 7/7 doğru |

## Veri seti

- **GTSRB** (6 sınıf): dur, yol_ver, girilmez, hiz_siniri_50, yaya_dikkat, sola_mecburi  
  https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
- **İnternetten toplanan Türk levhaları** (3 sınıf): sola_donulmez, yaya_gecidi, trafik_lambasi  
  DuckDuckGo görsel araması ile toplandı, elle temizlendi. Telif nedeniyle repoya eklenmedi.
- Train: 1044, validation: 262, test: 384 görüntü.

## Model ve eğitim

- Mimari: ResNet18 (ImageNet ağırlıkları, fine-tuning), girdi 224×224
- Optimizer: AdamW, learning rate 1×10⁻⁴, batch size 32, 15 epoch
- Loss: CrossEntropyLoss
- Augmentation: RandomResizedCrop, RandomRotation(15°), ColorJitter

## Kurulum

```
pip install -r requirements.txt
```

## Çalıştırma

1. `python download_data.py`: GTSRB'yi indirir ve `dataset/` klasörünü kurar. Ek sınıflar için `extra/<sinif_adi>/` klasörlerine görsel koyun.
2. `python train.py`: modeli eğitir, `model/resnet_traffic.pth` dosyasını kaydeder.
3. `python evaluate.py`: test metriklerini ve `test_images/` içindeki telefon fotoğraflarının sonuçlarını üretir.
4. `python app.py`: Gradio arayüzünü başlatır (http://127.0.0.1:7860).

## Dosya yapısı

```
app.py              Gradio arayüzü
train.py            Eğitim
evaluate.py         Değerlendirme ve gerçek fotoğraf testi
download_data.py    Veri hazırlama
collect_images.py   Ek sınıflar için görsel toplama
test_images/        Telefonla çekilen test fotoğrafları
model/              Eğitilmiş model
```

## Sınırlılıklar

- İnternet sınıflarının test kümesi çok küçüktür (6–10 görüntü), bu sınıfların metrikleri belirsizdir.
- Sınıf başına görüntü sayısı dengesizdir (200 / 27–44).
- Model tek levha içeren, kırpılmış fotoğraflar içindir. Bir karedeki birden fazla levha için nesne tespiti (ör. YOLO) gerekir.
- yaya_dikkat ve yaya_gecidi levhaları birbiriyle karışabilir."# traffic_signs" 
