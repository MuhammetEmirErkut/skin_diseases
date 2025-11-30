"""
Ana çalıştırma dosyası - Cilt Hastalıkları CNN Sınıflandırıcı
"""
import os
import sys

# Proje kök dizinini path'e ekle
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.train import ModelTrainer
from src.train_improved import ImprovedModelTrainer
from src.predict import SkinDiseasePredictor
from src.data_loader import SkinDiseaseDataLoader

def main():
    """Ana fonksiyon"""
    print("=" * 70)
    print(" " * 15 + "CİLT HASTALIKLARI CNN SINIFLANDIRICI")
    print(" " * 20 + "10 Sınıf Sınıflandırma Projesi")
    print("=" * 70)
    
    print("\n[INFO] Menu:")
    print("1. Veri setini kontrol et")
    print("2. Model egit (Transfer Learning)")
    print("3. Model egit (Basit CNN)")
    print("4. Model egit (Gelistirilmis - Notebook Yaklasimi)")
    print("5. Model ile tahmin yap")
    print("6. Cikis")
    
    choice = input("\nSeçiminiz (1-5): ").strip()
    
    if choice == '1':
        check_dataset()
    elif choice == '2':
        train_model(use_transfer=True)
    elif choice == '3':
        train_model(use_transfer=False)
    elif choice == '4':
        train_improved_model()
    elif choice == '5':
        predict_image()
    elif choice == '6':
        print("Cikiliyor...")
        return
    else:
        print("Gecersiz secim!")

def check_dataset():
    """Veri setini kontrol et"""
    print("\n" + "=" * 70)
    print("VERİ SETİ KONTROLÜ")
    print("=" * 70)
    
    loader = SkinDiseaseDataLoader()
    train_path = loader.find_dataset_path()
    
    if train_path:
        print(f"[OK] Veri seti bulundu: {train_path}")
        
        # Sınıfları kontrol et
        print("\n[INFO] Sinif kontrolu:")
        for i, class_name in enumerate(loader.SELECTED_CLASSES, 1):
            class_path = os.path.join(train_path, class_name)
            if os.path.exists(class_path):
                img_count = len([f for f in os.listdir(class_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"  {i}. {class_name}: {img_count} goruntu [OK]")
            else:
                print(f"  {i}. {class_name}: BULUNAMADI [EKSIK]")
    else:
        print("[HATA] Veri seti bulunamadi!")
        print("\nLütfen önce download_dataset.py scriptini çalıştırın:")
        print("  python download_dataset.py")

def train_model(use_transfer=True):
    """Model eğit"""
    print("\n" + "=" * 70)
    print("MODEL EĞİTİMİ")
    print("=" * 70)
    
    if use_transfer:
        print("\nModel tipi: Transfer Learning (EfficientNetB0)")
        base_model = input("Base model seçin (efficientnet/resnet/mobilenet) [efficientnet]: ").strip().lower()
        if not base_model:
            base_model = 'efficientnet'
    else:
        print("\nModel tipi: Basit CNN")
        base_model = 'simple'
    
    # Parametreler
    batch_size = input("Batch size [32]: ").strip()
    batch_size = int(batch_size) if batch_size else 32
    
    epochs = input("Epoch sayısı [50]: ").strip()
    epochs = int(epochs) if epochs else 50
    
    # Eğitici oluştur
    trainer = ModelTrainer(
        model_type='transfer' if use_transfer else 'simple',
        base_model=base_model if use_transfer else 'simple',
        img_size=(224, 224),
        batch_size=batch_size,
        epochs=epochs
    )
    
    try:
        # Veriyi yükle
        trainer.load_data()
        
        # Modeli oluştur
        trainer.build_model(num_classes=10)
        
        # Eğit
        trainer.train()
        
        # Değerlendir
        trainer.evaluate()
        
        # Görselleştir
        trainer.plot_history()
        
        # Kaydet
        trainer.save_results()
        
        print("\n" + "=" * 70)
        print("[OK] EGITIM TAMAMLANDI!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[HATA] Hata: {e}")
        import traceback
        traceback.print_exc()

def train_improved_model():
    """Geliştirilmiş model eğit (Notebook yaklaşımı)"""
    print("\n" + "=" * 70)
    print("GELİŞTİRİLMİŞ MODEL EĞİTİMİ")
    print("=" * 70)
    print("\nBu yontem Notebook'taki basarili yaklasimi kullanir:")
    print("  - Adamax optimizer (Notebook'taki gibi)")
    print("  - Class weights KAPALI (Notebook'ta yok)")
    print("  - Daha esnek early stopping")
    print("  - Yavas learning rate reduction")
    
    print("\nModel tipi secin:")
    print("1. Transfer Learning (Onerilen)")
    print("2. Basit CNN")
    model_choice = input("Seciminiz (1/2) [1]: ").strip()
    
    if model_choice == '2':
        model_type = 'simple'
        base_model = 'simple'
    else:
        model_type = 'transfer'
        print("\nBase model secin:")
        print("1. EfficientNetB0 (Onerilen)")
        print("2. ResNet50")
        print("3. MobileNetV2")
        base_choice = input("Seciminiz (1/2/3) [1]: ").strip()
        
        if base_choice == '2':
            base_model = 'resnet'
        elif base_choice == '3':
            base_model = 'mobilenet'
        else:
            base_model = 'efficientnet'
    
    # Parametreler
    batch_size = input("Batch size [32]: ").strip()
    batch_size = int(batch_size) if batch_size else 32
    
    epochs = input("Epoch sayisi [50]: ").strip()
    epochs = int(epochs) if epochs else 50
    
    # Geliştirilmiş eğitici oluştur
    trainer = ImprovedModelTrainer(
        model_type=model_type,
        base_model=base_model if model_type == 'transfer' else 'simple',
        img_size=(224, 224),
        batch_size=batch_size,
        epochs=epochs,
        use_class_weights=False  # Notebook'ta yok
    )
    
    try:
        # Veriyi yükle
        trainer.load_data()
        
        # Modeli oluştur
        trainer.build_model(num_classes=10)
        
        # Eğit
        trainer.train()
        
        # Değerlendir
        trainer.evaluate()
        
        # Görselleştir
        trainer.plot_history()
        
        # Kaydet
        trainer.save_results()
        
        print("\n" + "=" * 70)
        print("[OK] EGITIM TAMAMLANDI!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[HATA] Hata: {e}")
        import traceback
        traceback.print_exc()

def predict_image():
    """Görüntü ile tahmin yap"""
    print("\n" + "=" * 70)
    print("TAHMIN YAPMA")
    print("=" * 70)
    
    # Model dosyasını bul
    import glob
    from datetime import datetime
    
    model_pattern = 'models/best_model_*.h5'
    models = glob.glob(model_pattern)
    
    if not models:
        print("[UYARI] Model bulunamadi! Lutfen once egitim yapin.")
        return
    
    # Modelleri tarih/saat sırasına göre sırala (en yeni en üstte)
    def extract_timestamp(model_path):
        """Dosya isminden timestamp çıkar"""
        filename = os.path.basename(model_path)
        # Format: best_model_TYPE_BASE_YYYYMMDD_HHMMSS.h5
        try:
            parts = filename.replace('.h5', '').split('_')
            if len(parts) >= 6:
                date_str = parts[-2]  # YYYYMMDD
                time_str = parts[-1]  # HHMMSS
                timestamp_str = f"{date_str}_{time_str}"
                return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except:
            # Eğer parse edilemezse, dosya değişiklik zamanını kullan
            return datetime.fromtimestamp(os.path.getmtime(model_path))
        return datetime.min
    
    models_sorted = sorted(models, key=extract_timestamp, reverse=True)
    
    print("\nMevcut modeller (en yeni en üstte):")
    for i, model in enumerate(models_sorted, 1):
        model_name = os.path.basename(model)
        timestamp = extract_timestamp(model)
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        if i == 1:
            print(f"  {i}. {model_name} [EN SON] - {timestamp_str}")
        else:
            print(f"  {i}. {model_name} - {timestamp_str}")
    
    model_idx = input(f"\nModel secin (1-{len(models_sorted)}) [1]: ").strip()
    model_idx = int(model_idx) - 1 if model_idx else 0
    
    if model_idx < 0 or model_idx >= len(models_sorted):
        print("Gecersiz secim!")
        return
    
    model_path = models_sorted[model_idx]
    
    # Görüntü seçimi
    print("\n" + "=" * 70)
    print("GORUNTU SECIMI")
    print("=" * 70)
    print("\n1. Test klasorunden sec (Onerilen)")
    print("2. Manuel dosya yolu gir")
    
    choice = input("\nSeciminiz (1/2) [1]: ").strip()
    
    if choice == '2':
        # Manuel yol girme
        print("\n[INFO] Ornek dosya yollari:")
        print("  Dataset/test/Acne and Rosacea Photos/acne-cystic-33.jpg")
        print("  Dataset/test/Melanoma Skin Cancer Nevi and Moles/malignant-melanoma-177.jpg")
        print("  Dataset/test/Eczema Photos/eczema-subacute-61.jpg")
        
        image_path = input("\nGoruntu dosyasi yolu: ").strip().strip('"').strip("'")
        
        # Dosya yolu düzeltmeleri
        if not os.path.exists(image_path):
            possible_paths = [
                image_path,
                os.path.join('Dataset', 'test', image_path),
                os.path.join('Dataset', 'train', image_path),
                image_path.replace('./', ''),
                image_path.replace('.\\', ''),
            ]
            
            if '/' not in image_path and '\\' not in image_path:
                test_dir = 'Dataset/test'
                if os.path.exists(test_dir):
                    for root, dirs, files in os.walk(test_dir):
                        if image_path in files:
                            possible_paths.append(os.path.join(root, image_path))
                            break
            
            found = False
            for path in possible_paths:
                if os.path.exists(path):
                    image_path = path
                    print(f"[OK] Dosya bulundu: {image_path}")
                    found = True
                    break
            
            if not found:
                print(f"[HATA] Dosya bulunamadi: {image_path}")
                return
    else:
        # Test klasöründen seç
        image_path = select_from_test_folder()
        if not image_path:
            return
    
    try:
        # Predictor oluştur
        predictor = SkinDiseasePredictor(model_path)
        
        # Tahmin yap ve görselleştir (tüm 10 sınıfı göster)
        predictor.visualize_prediction(image_path, show_all_classes=True)
        
    except Exception as e:
        print(f"\n[HATA] Hata: {e}")
        import traceback
        traceback.print_exc()

def select_from_test_folder():
    """Test klasöründen görüntü seç"""
    test_dir = 'Dataset/test'
    
    if not os.path.exists(test_dir):
        print(f"[HATA] Test klasoru bulunamadi: {test_dir}")
        return None
    
    # Sınıfları listele
    loader = SkinDiseaseDataLoader()
    available_classes = []
    
    print("\n[INFO] Mevcut siniflar:")
    for i, class_name in enumerate(loader.SELECTED_CLASSES, 1):
        class_path = os.path.join(test_dir, class_name)
        if os.path.exists(class_path):
            img_files = [f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if img_files:
                available_classes.append((class_name, class_path, img_files))
                print(f"  {i}. {class_name} ({len(img_files)} goruntu)")
    
    if not available_classes:
        print("[HATA] Test klasorunde goruntu bulunamadi!")
        return None
    
    # Sınıf seç
    class_choice = input(f"\nSinif secin (1-{len(available_classes)}) [1]: ").strip()
    try:
        class_idx = int(class_choice) - 1 if class_choice else 0
        if class_idx < 0 or class_idx >= len(available_classes):
            class_idx = 0
    except:
        class_idx = 0
    
    class_name, class_path, img_files = available_classes[class_idx]
    
    # Görüntüleri listele (ilk 20'yi göster)
    print(f"\n[INFO] {class_name} sinifindaki goruntuler:")
    display_count = min(20, len(img_files))
    
    for i in range(display_count):
        print(f"  {i+1}. {img_files[i]}")
    
    if len(img_files) > display_count:
        print(f"  ... ve {len(img_files) - display_count} goruntu daha")
    
    # Görüntü seç
    img_choice = input(f"\nGoruntu secin (1-{len(img_files)}) [1]: ").strip()
    try:
        img_idx = int(img_choice) - 1 if img_choice else 0
        if img_idx < 0 or img_idx >= len(img_files):
            img_idx = 0
    except:
        img_idx = 0
    
    selected_image = img_files[img_idx]
    image_path = os.path.join(class_path, selected_image)
    
    print(f"\n[OK] Secilen goruntu: {image_path}")
    return image_path

if __name__ == "__main__":
    main()

