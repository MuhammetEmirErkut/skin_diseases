"""
Batch Size Öneri Aracı
GPU/CPU durumunu kontrol edip optimal batch size önerir
"""
import os
import sys
import tensorflow as tf

def check_gpu():
    """GPU durumunu kontrol et"""
    print("=" * 70)
    print(" " * 20 + "SISTEM DURUMU KONTROLU")
    print("=" * 70)
    
    # TensorFlow GPU kontrolü
    print("\n[INFO] TensorFlow GPU Kontrolu:")
    gpus = tf.config.list_physical_devices('GPU')
    
    if len(gpus) > 0:
        print(f"[OK] {len(gpus)} GPU bulundu!")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                print(f"    Detaylar: {gpu_details}")
            except:
                pass
        
        # GPU bellek bilgisi
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("[OK] GPU bellek buyumesi aktif")
        except:
            pass
        
        return True, "gpu"
    else:
        print("[UYARI] GPU bulunamadi, CPU kullanilacak")
        return False, "cpu"

def get_recommended_batch_size(has_gpu, model_type, base_model, img_size):
    """Önerilen batch size'ı hesapla"""
    print("\n" + "=" * 70)
    print(" " * 20 + "BATCH SIZE ONERILERI")
    print("=" * 70)
    
    # Temel öneriler
    recommendations = {
        'gpu': {
            'transfer': {
                'efficientnet': {'min': 16, 'optimal': 32, 'max': 64},
                'resnet': {'min': 16, 'optimal': 32, 'max': 64},
                'mobilenet': {'min': 32, 'optimal': 64, 'max': 128}
            },
            'simple': {
                'default': {'min': 16, 'optimal': 32, 'max': 64}
            }
        },
        'cpu': {
            'transfer': {
                'efficientnet': {'min': 4, 'optimal': 8, 'max': 16},
                'resnet': {'min': 4, 'optimal': 8, 'max': 16},
                'mobilenet': {'min': 8, 'optimal': 16, 'max': 32}
            },
            'simple': {
                'default': {'min': 4, 'optimal': 8, 'max': 16}
            }
        }
    }
    
    device = 'gpu' if has_gpu else 'cpu'
    
    if model_type == 'transfer':
        if base_model in recommendations[device]['transfer']:
            rec = recommendations[device]['transfer'][base_model]
        else:
            rec = recommendations[device]['transfer']['efficientnet']
    else:
        rec = recommendations[device]['simple']['default']
    
    print(f"\n[INFO] Model Tipi: {model_type.upper()}")
    if model_type == 'transfer':
        print(f"[INFO] Base Model: {base_model.upper()}")
    print(f"[INFO] Goruntu Boyutu: {img_size}")
    print(f"[INFO] Cihaz: {device.upper()}")
    
    print(f"\n[ONERI] Batch Size Secenekleri:")
    print(f"  Minimum: {rec['min']}  (Bellek yetersizse)")
    print(f"  Optimal: {rec['optimal']}  (Onerilen)")
    print(f"  Maksimum: {rec['max']}  (Bellek yeterliyse)")
    
    # Veri seti boyutuna göre öneri
    print(f"\n[INFO] Veri Seti: 1579 goruntu")
    print(f"[INFO] Optimal batch size ile:")
    print(f"  - Epoch basina ~{1579 // rec['optimal']} batch")
    print(f"  - Her batch ~{rec['optimal']} goruntu")
    
    return rec

def test_batch_size(batch_size, model_type, base_model, img_size):
    """Batch size'ı test et (opsiyonel)"""
    print(f"\n[INFO] Batch size {batch_size} test ediliyor...")
    
    try:
        # Küçük bir test modeli oluştur
        if model_type == 'transfer':
            from tensorflow.keras.applications import EfficientNetB0, ResNet50, MobileNetV2
            
            if base_model == 'efficientnet':
                base = EfficientNetB0(weights=None, include_top=False, input_shape=(*img_size, 3))
            elif base_model == 'resnet':
                base = ResNet50(weights=None, include_top=False, input_shape=(*img_size, 3))
            else:
                base = MobileNetV2(weights=None, include_top=False, input_shape=(*img_size, 3))
        else:
            from tensorflow.keras import Sequential, layers
            base = Sequential([
                layers.Conv2D(32, (3, 3), input_shape=(*img_size, 3)),
                layers.GlobalAveragePooling2D()
            ])
        
        # Test verisi oluştur
        import numpy as np
        test_data = np.random.random((batch_size, *img_size, 3))
        
        # Forward pass testi
        _ = base(test_data)
        print(f"[OK] Batch size {batch_size} calisiyor!")
        return True
        
    except Exception as e:
        print(f"[HATA] Batch size {batch_size} bellek hatasi: {e}")
        return False

def main():
    """Ana fonksiyon"""
    # GPU kontrolü
    has_gpu, device = check_gpu()
    
    # Model tipi seçimi
    print("\n" + "=" * 70)
    print("Model tipi secin:")
    print("1. Transfer Learning")
    print("2. Basit CNN")
    choice = input("Seciminiz (1/2) [1]: ").strip()
    
    if choice == '2':
        model_type = 'simple'
        base_model = 'default'
    else:
        model_type = 'transfer'
        print("\nBase model secin:")
        print("1. EfficientNetB0 (Onerilen)")
        print("2. ResNet50")
        print("3. MobileNetV2")
        model_choice = input("Seciminiz (1/2/3) [1]: ").strip()
        
        if model_choice == '2':
            base_model = 'resnet'
        elif model_choice == '3':
            base_model = 'mobilenet'
        else:
            base_model = 'efficientnet'
    
    # Görüntü boyutu
    img_size = (224, 224)
    
    # Önerileri al
    recommendations = get_recommended_batch_size(
        has_gpu, model_type, base_model, img_size
    )
    
    # Kullanıcı seçimi
    print("\n" + "=" * 70)
    print("Batch size secin:")
    print(f"  [1] Minimum: {recommendations['min']}")
    print(f"  [2] Optimal: {recommendations['optimal']} (Onerilen)")
    print(f"  [3] Maksimum: {recommendations['max']}")
    print(f"  [4] Ozel deger girin")
    
    batch_choice = input("\nSeciminiz (1/2/3/4) [2]: ").strip()
    
    if batch_choice == '1':
        selected_batch = recommendations['min']
    elif batch_choice == '3':
        selected_batch = recommendations['max']
    elif batch_choice == '4':
        try:
            selected_batch = int(input("Batch size girin: ").strip())
        except:
            selected_batch = recommendations['optimal']
    else:
        selected_batch = recommendations['optimal']
    
    print(f"\n[OK] Secilen batch size: {selected_batch}")
    
    # Test etmek ister misiniz?
    test = input("\nBatch size'i test etmek ister misiniz? (e/h) [h]: ").strip().lower()
    if test == 'e':
        test_batch_size(selected_batch, model_type, base_model, img_size)
    
    print("\n" + "=" * 70)
    print("KULLANIM:")
    print("=" * 70)
    print(f"\nPython kodunda:")
    print(f"  trainer = ModelTrainer(")
    print(f"      model_type='{model_type}',")
    if model_type == 'transfer':
        print(f"      base_model='{base_model}',")
    print(f"      batch_size={selected_batch},")
    print(f"      epochs=50")
    print(f"  )")
    
    print(f"\nVeya main.py'de batch size soruldugunda: {selected_batch}")

if __name__ == "__main__":
    main()




