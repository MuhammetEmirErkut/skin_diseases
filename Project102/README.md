# Cilt Hastalıkları CNN Sınıflandırıcı

10 temel cilt hastalığını görüntülerden sınıflandıran derin öğrenme projesi.

## 📋 Proje Hakkında

Bu proje, görüntü işleme ve derin öğrenme teknikleri kullanarak 10 farklı cilt hastalığını sınıflandırmayı amaçlar. Transfer Learning ve gelişmiş CNN mimarileri kullanılmıştır.

### Sınıflandırılan 10 Cilt Hastalığı

1. Acne and Rosacea Photos (Akne ve Rozasea)
2. Atopic Dermatitis Photos (Atopik Dermatit)
3. Eczema Photos (Egzama)
4. Melanoma Skin Cancer Nevi and Moles (Melanom)
5. Psoriasis pictures Lichen Planus and related diseases (Sedef Hastalığı)
6. Tinea Ringworm Candidiasis and other Fungal Infections (Mantar Enfeksiyonları)
7. Urticaria Hives (Kurdeşen)
8. Warts Molluscum and other Viral Infections (Siğil ve Viral Enfeksiyonlar)
9. Seborrheic Keratoses and other Benign Tumors (Seboreik Keratoz)
10. Cellulitis Impetigo and other Bacterial Infections (Selülit ve Bakteriyel Enfeksiyonlar)

## 🚀 Kurulum

### 1. Gereksinimler

```bash
pip install -r requirements.txt
```

### 2. Veri Setini İndirme (Manuel - API Gerektirmez)

**Yöntem 1: Otomatik Kontrol ve Çıkarma**

```bash
python download_dataset.py
```

Bu script:
- Mevcut veri setini kontrol eder
- Zip dosyası varsa otomatik çıkarır
- Veri seti yapısını doğrular
- Eksikse manuel indirme talimatları verir

**Yöntem 2: Manuel İndirme**

1. Tarayıcınızda şu adrese gidin:
   ```
   https://www.kaggle.com/datasets/shreyas1720/20-skin-diseases-dataset
   ```

2. Sayfanın sağ üstünde **"Download"** butonuna tıklayın
   - Kaggle hesabı gerektirebilir (ücretsiz kayıt olabilirsiniz)

3. İndirilen `20-skin-diseases-dataset.zip` dosyasını proje klasörüne kopyalayın

4. Zip dosyasını çıkarın:
   - **Windows**: Sağ tık → "Extract All" / "Tümünü Çıkar"
   - **Linux/Mac**: `unzip 20-skin-diseases-dataset.zip`
   - Çıkarma hedefi: Proje klasörü (Dataset/ klasörü oluşturulmalı)

5. Scripti tekrar çalıştırın:
   ```bash
   python download_dataset.py
   ```

**Not**: Kaggle Notebook ortamında çalışıyorsanız, veri seti zaten `/kaggle/input/` dizininde mevcut olabilir.

## 📁 Proje Yapısı

```
Project102/
├── download_dataset.py      # Veri seti indirme scripti
├── main.py                  # Ana çalıştırma dosyası
├── requirements.txt         # Python paketleri
├── README.md               # Bu dosya
├── src/
│   ├── data_loader.py      # Veri yükleme modülü
│   ├── model.py            # Model tanımlamaları
│   ├── train.py            # Eğitim scripti
│   └── predict.py          # Tahmin scripti
├── Dataset/                 # Veri seti (indirme sonrası)
│   ├── train/
│   └── test/
├── models/                  # Eğitilmiş modeller
├── logs/                    # Eğitim logları
└── results/                 # Sonuçlar ve görselleştirmeler
```

## 🎯 Kullanım

### Yöntem 1: İnteraktif Menü

```bash
python main.py
```

Menüden seçim yapın:
1. Veri setini kontrol et
2. Model eğit (Transfer Learning)
3. Model eğit (Basit CNN)
4. Model ile tahmin yap

### Yöntem 2: Doğrudan Eğitim

```bash
python src/train.py
```

Veya Python'da:

```python
from src.train import ModelTrainer

trainer = ModelTrainer(
    model_type='transfer',
    base_model='efficientnet',
    img_size=(224, 224),
    batch_size=32,
    epochs=50
)

trainer.load_data()
trainer.build_model(num_classes=10)
trainer.train()
trainer.evaluate()
trainer.plot_history()
```

### Yöntem 3: Tahmin Yapma

```python
from src.predict import SkinDiseasePredictor

predictor = SkinDiseasePredictor('models/best_model_*.h5')
results = predictor.predict('path/to/image.jpg', top_k=3)
predictor.visualize_prediction('path/to/image.jpg')
```

## 🏗️ Model Mimarileri

### 1. Transfer Learning (Önerilen)

- **EfficientNetB0**: En iyi performans
- **ResNet50**: Dengeli performans
- **MobileNetV2**: Hızlı ve hafif

### 2. Basit CNN

Notebook'taki modelden esinlenilmiş ancak geliştirilmiş:
- Batch Normalization
- Dropout katmanları
- Gelişmiş data augmentation

## 📊 Model Performansı

Eğitim sonrası şu metrikler kaydedilir:
- Training/Validation Accuracy
- Training/Validation Loss
- Top-3 Accuracy
- Confusion Matrix
- Classification Report

Sonuçlar `results/` klasöründe kaydedilir.

## 🔧 Özelleştirme

### Model Parametrelerini Değiştirme

`src/train.py` dosyasında:

```python
trainer = ModelTrainer(
    model_type='transfer',      # 'transfer' veya 'simple'
    base_model='efficientnet',   # 'efficientnet', 'resnet', 'mobilenet'
    img_size=(224, 224),        # Görüntü boyutu
    batch_size=32,              # Batch boyutu
    epochs=50                   # Epoch sayısı
)
```

### Veri Yükleme Parametrelerini Değiştirme

`src/data_loader.py` dosyasında:

```python
loader = SkinDiseaseDataLoader(img_size=(224, 224))
```

## 📝 Notlar

- **GPU Önerilir**: Eğitim süresi GPU ile önemli ölçüde azalır
- **Bellek**: En az 8GB RAM önerilir
- **Disk Alanı**: Veri seti için ~500MB alan gerekir
- **Eğitim Süresi**: GPU ile ~30-60 dakika, CPU ile birkaç saat

## 🐛 Sorun Giderme

### Veri Seti Bulunamadı

```bash
# Veri setini kontrol edin
python -c "from src.data_loader import SkinDiseaseDataLoader; loader = SkinDiseaseDataLoader(); print(loader.find_dataset_path())"

# Eğer None dönerse, download_dataset.py çalıştırın
python download_dataset.py
```

### Kaggle API Hatası

- `kaggle.json` dosyasının doğru konumda olduğundan emin olun
- Dosya izinlerini kontrol edin (Linux/Mac: `chmod 600 ~/.kaggle/kaggle.json`)

### CUDA/GPU Hatası

- TensorFlow GPU sürümünü yükleyin: `pip install tensorflow-gpu`
- CUDA ve cuDNN'in doğru yüklendiğinden emin olun

## 📚 Referanslar

- [Kaggle Dataset](https://www.kaggle.com/datasets/shreyas1720/20-skin-diseases-dataset)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)

## 📄 Lisans

Bu proje eğitim amaçlıdır.

## 👤 Yazar

Proje, mevcut Kaggle notebook'larından esinlenilerek geliştirilmiştir.

---

**Not**: Bu proje tıbbi tanı amaçlı değildir. Sadece eğitim ve araştırma amaçlıdır.

