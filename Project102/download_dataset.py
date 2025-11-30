"""
Veri seti kurulum scripti - Manuel indirme için
Kaggle API kullanmadan çalışır
"""
import os
import sys
import zipfile
from pathlib import Path

# Windows konsolu için encoding ayarla
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def check_dataset():
    """Veri setinin mevcut olup olmadığını kontrol et"""
    possible_paths = [
        'Dataset/train/',
        '../input/20-skin-diseases-dataset/Dataset/train/',
        '/kaggle/input/20-skin-diseases-dataset/Dataset/train/',
        './Dataset/train/'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"[OK] Veri seti bulundu: {path}")
            return path
    
    return None

def check_zip_file():
    """Zip dosyasının mevcut olup olmadığını kontrol et"""
    zip_files = [
        '20-skin-diseases-dataset.zip',
        'Dataset.zip',
        'skin-diseases-dataset.zip'
    ]
    
    for zip_file in zip_files:
        if os.path.exists(zip_file):
            print(f"[OK] Zip dosyası bulundu: {zip_file}")
            return zip_file
    
    return None

def extract_zip(zip_path, extract_to="Dataset"):
    """Zip dosyasını çıkar"""
    if not os.path.exists(zip_path):
        print(f"[HATA] {zip_path} dosyası bulunamadı!")
        return False
    
    print(f"\n[INFO] Veri seti cikariliyor: {extract_to}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"[OK] Veri seti basariyla cikarildi: {extract_to}")
        return True
    except Exception as e:
        print(f"[HATA] Cikarma hatasi: {e}")
        return False

def verify_dataset_structure(dataset_path):
    """Veri seti yapısını doğrula"""
    required_classes = [
        'Acne and Rosacea Photos',
        'Atopic Dermatitis Photos',
        'Eczema Photos',
        'Melanoma Skin Cancer Nevi and Moles',
        'Psoriasis pictures Lichen Planus and related diseases',
        'Tinea Ringworm Candidiasis and other Fungal Infections',
        'Urticaria Hives',
        'Warts Molluscum and other Viral Infections',
        'Seborrheic Keratoses and other Benign Tumors',
        'Cellulitis Impetigo and other Bacterial Infections'
    ]
    
    print("\n[INFO] Veri seti yapisi kontrol ediliyor...")
    found_classes = []
    missing_classes = []
    
    for class_name in required_classes:
        class_path = os.path.join(dataset_path, class_name)
        if os.path.exists(class_path):
            img_count = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            found_classes.append((class_name, img_count))
            print(f"  [OK] {class_name}: {img_count} goruntu")
        else:
            missing_classes.append(class_name)
            print(f"  [EKSIK] {class_name}: BULUNAMADI")
    
    if missing_classes:
        print(f"\n[UYARI] {len(missing_classes)} sinif bulunamadi!")
        return False
    
    total_images = sum(count for _, count in found_classes)
    print(f"\n[OK] Tum siniflar mevcut! Toplam {total_images} goruntu bulundu.")
    return True

def main():
    """Ana fonksiyon"""
    print("=" * 70)
    print(" " * 15 + "VERİ SETİ KURULUM KONTROLÜ")
    print("=" * 70)
    
    # Önce mevcut veri setini kontrol et
    dataset_path = check_dataset()
    if dataset_path:
        if verify_dataset_structure(dataset_path):
            print("\n" + "=" * 70)
            print("[OK] Veri seti hazir! Projeyi calistirabilirsiniz.")
            print("=" * 70)
            print("\nSonraki adim:")
            print("  python main.py")
            return
        else:
            print("\n[UYARI] Veri seti eksik gorunuyor.")
    
    # Zip dosyasını kontrol et
    zip_path = check_zip_file()
    if zip_path:
        print(f"\n[INFO] Zip dosyasi bulundu: {zip_path}")
        response = input("Zip dosyasini cikarmak ister misiniz? (e/h) [e]: ").strip().lower()
        if response != 'h':
            if extract_zip(zip_path):
                dataset_path = check_dataset()
                if dataset_path:
                    verify_dataset_structure(dataset_path)
                print("\n[OK] Veri seti hazir!")
                return
    
    # Veri seti bulunamadı - manuel indirme talimatları
    print("\n" + "=" * 70)
    print(" " * 20 + "MANUEL INDIRME TALIMATLARI")
    print("=" * 70)
    print("\nVeri seti bulunamadı. Lütfen şu adımları izleyin:\n")
    print("1. Tarayıcınızda şu adrese gidin:")
    print("   https://www.kaggle.com/datasets/shreyas1720/20-skin-diseases-dataset")
    print("\n2. Sayfanın sağ üstünde 'Download' butonuna tıklayın")
    print("   (Kaggle hesabı gerektirebilir - ücretsiz kayıt olabilirsiniz)")
    print("\n3. İndirilen zip dosyasını bu proje klasörüne kopyalayın")
    print("   Dosya adı: '20-skin-diseases-dataset.zip' olmalı")
    print("\n4. Bu scripti tekrar çalıştırın:")
    print("   python download_dataset.py")
    print("\n" + "=" * 70)
    print("\nAlternatif: Zip dosyasını manuel olarak çıkarmak için:")
    print("1. '20-skin-diseases-dataset.zip' dosyasını bulun")
    print("2. Sağ tıklayın → 'Extract All' veya 'Tümünü Çıkar'")
    print("3. Çıkarma hedefi: Bu proje klasörü")
    print("4. 'Dataset' klasörü oluşturulmalı")
    print("\n" + "=" * 70)
    
    # Mevcut dizindeki dosyaları göster
    print("\n[INFO] Mevcut dizindeki dosyalar:")
    current_dir = Path('.')
    files = [f.name for f in current_dir.iterdir() if f.is_file()]
    dirs = [d.name for d in current_dir.iterdir() if d.is_dir()]
    
    if files:
        print("  Dosyalar:")
        for f in files[:10]:  # İlk 10 dosyayı göster
            print(f"    - {f}")
        if len(files) > 10:
            print(f"    ... ve {len(files) - 10} dosya daha")
    
    if dirs:
        print("  Klasörler:")
        for d in dirs:
            print(f"    - {d}/")

if __name__ == "__main__":
    main()
