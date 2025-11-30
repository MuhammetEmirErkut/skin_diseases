"""
Veri yükleme ve ön işleme modülü
"""
import os
import sys
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional

# Windows konsolu için encoding ayarla
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

class SkinDiseaseDataLoader:
    """Cilt hastalığı verilerini yükleyen ve ön işleyen sınıf"""
    
    # 10 temel cilt hastalığı sınıfı
    SELECTED_CLASSES = [
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
    
    def __init__(self, img_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            img_size: Görüntü boyutu (height, width)
        """
        self.img_size = img_size
        self.num_classes = len(self.SELECTED_CLASSES)
        self.images = []
        self.labels = []
        
    def find_dataset_path(self) -> Optional[str]:
        """Veri seti yolunu bul (Kaggle veya yerel)"""
        # Kaggle ortamı
        kaggle_path = '/kaggle/input/20-skin-diseases-dataset/Dataset/train/'
        if os.path.exists(kaggle_path):
            return kaggle_path
        
        # Yerel ortam
        local_paths = [
            'Dataset/train/',
            '../input/20-skin-diseases-dataset/Dataset/train/',
            './Dataset/train/'
        ]
        
        for path in local_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def load_images(self, train_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Görüntüleri yükle ve ön işle
        
        Returns:
            images: Normalize edilmiş görüntü dizisi
            labels: One-hot encoded etiket dizisi
        """
        if train_path is None:
            train_path = self.find_dataset_path()
        
        if train_path is None:
            raise FileNotFoundError(
                "Veri seti bulunamadı! Lütfen download_dataset.py scriptini çalıştırın."
            )
        
        print(f"[INFO] Veri yolu: {train_path}")
        print(f"[INFO] {self.num_classes} sinif yukleniyor...\n")
        
        self.images = []
        self.labels = []
        
        for idx, class_name in enumerate(self.SELECTED_CLASSES):
            class_path = os.path.join(train_path, class_name)
            
            if not os.path.exists(class_path):
                print(f"[UYARI] {class_name} klasoru bulunamadi, atlaniyor...")
                continue
            
            print(f"[{idx+1}/{self.num_classes}] {class_name}", end=" -> ")
            img_count = 0
            
            # Klasördeki tüm görüntü dosyalarını işle
            for img_file in os.listdir(class_path):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                
                img_path = os.path.join(class_path, img_file)
                
                try:
                    # Görüntüyü oku
                    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if image is None:
                        continue
                    
                    # BGR'den RGB'ye çevir
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Boyutlandır
                    image = cv2.resize(image, self.img_size)
                    
                    # Normalize et (0-1 aralığına)
                    image = image.astype(np.float32) / 255.0
                    
                    self.images.append(image)
                    
                    # One-hot encoding label
                    label = np.zeros(self.num_classes, dtype=np.float32)
                    label[idx] = 1.0
                    self.labels.append(label)
                    
                    img_count += 1
                    
                except Exception as e:
                    print(f"\n   Hata ({img_file}): {e}")
                    continue
            
            print(f"{img_count} goruntu [OK]")
        
        # NumPy array'e çevir
        images_array = np.array(self.images, dtype=np.float32)
        labels_array = np.array(self.labels, dtype=np.float32)
        
        print(f"\n[OK] Toplam {len(self.images)} goruntu yuklendi")
        print(f"[OK] Goruntu boyutu: {images_array.shape}")
        print(f"[OK] Etiket boyutu: {labels_array.shape}")
        
        return images_array, labels_array
    
    def split_data(
        self, 
        images: np.ndarray, 
        labels: np.ndarray,
        test_size: float = 0.15,  # Notebook'taki gibi %15 test
        val_size: float = 0.15,   # Notebook'taki gibi %15 validation
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Veriyi eğitim, doğrulama ve test setlerine ayır
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Önce train+val ve test'e ayır
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            images, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
        
        # Sonra train ve val'e ayır
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_size / (1 - test_size),  # Düzeltilmiş oran
            random_state=random_state,
            stratify=y_train_val
        )
        
        print(f"\n[INFO] Veri Bolunmesi:")
        print(f"  Egitim: {X_train.shape[0]} goruntu ({X_train.shape[0]/len(images)*100:.1f}%)")
        print(f"  Dogrulama: {X_val.shape[0]} goruntu ({X_val.shape[0]/len(images)*100:.1f}%)")
        print(f"  Test: {X_test.shape[0]} goruntu ({X_test.shape[0]/len(images)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_class_names(self) -> List[str]:
        """Sınıf isimlerini döndür"""
        return self.SELECTED_CLASSES.copy()

