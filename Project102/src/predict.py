"""
Tahmin ve değerlendirme scripti
"""
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from typing import List, Tuple, Optional

# Windows konsolu için encoding ayarla
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

try:
    from .data_loader import SkinDiseaseDataLoader
except ImportError:
    from data_loader import SkinDiseaseDataLoader

class SkinDiseasePredictor:
    """Cilt hastalığı tahmin sınıfı"""
    
    def __init__(self, model_path: str, img_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            model_path: Eğitilmiş model dosyası yolu
            img_size: Görüntü boyutu
        """
        self.model_path = model_path
        self.img_size = img_size
        self.model = None
        self.class_names = SkinDiseaseDataLoader.SELECTED_CLASSES
        
        self.load_model()
    
    def load_model(self):
        """Modeli yükle"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {self.model_path}")
        
        print(f"[INFO] Model yukleniyor: {self.model_path}")
        self.model = keras.models.load_model(self.model_path)
        print("[OK] Model yuklendi")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Tek bir görüntüyü ön işle
        
        Args:
            image_path: Görüntü dosyası yolu
            
        Returns:
            Ön işlenmiş görüntü array'i
        """
        # Görüntüyü oku
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Görüntü okunamadı: {image_path}")
        
        # BGR'den RGB'ye çevir
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Boyutlandır
        image = cv2.resize(image, self.img_size)
        
        # Normalize et
        image = image.astype(np.float32) / 255.0
        
        # Batch dimension ekle
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image_path: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Görüntü için tahmin yap
        
        Args:
            image_path: Görüntü dosyası yolu
            top_k: En yüksek k tahmin
            
        Returns:
            [(sınıf_adı, olasılık), ...] listesi
        """
        # Görüntüyü ön işle
        image = self.preprocess_image(image_path)
        
        # Tahmin yap
        predictions = self.model.predict(image, verbose=0)[0]
        
        # Top-k sonuçları al
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = [
            (self.class_names[idx], float(predictions[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def predict_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Birden fazla görüntü için toplu tahmin
        
        Args:
            image_paths: Görüntü dosyası yolları listesi
            
        Returns:
            Tahmin matrisi (n_images, n_classes)
        """
        images = []
        for img_path in image_paths:
            img = self.preprocess_image(img_path)
            images.append(img[0])  # Batch dimension'ı kaldır
        
        images = np.array(images)
        predictions = self.model.predict(images, verbose=0)
        
        return predictions
    
    def visualize_prediction(
        self,
        image_path: str,
        save_path: Optional[str] = None,
        show_all_classes: bool = True
    ):
        """
        Tahmin sonucunu görselleştir
        
        Args:
            image_path: Görüntü dosyası yolu
            save_path: Kaydetme yolu (opsiyonel)
            show_all_classes: Tüm sınıfları göster (True) veya sadece top-3 (False)
        """
        # Tahmin yap - tüm sınıflar için
        image = self.preprocess_image(image_path)
        predictions = self.model.predict(image, verbose=0)[0]
        
        # Tüm sınıfları sırala
        all_results = [
            (self.class_names[i], float(predictions[i]))
            for i in range(len(self.class_names))
        ]
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        # Gösterilecek sonuçları seç
        if show_all_classes:
            results = all_results
            title_suffix = " (Tum 10 Sinif)"
        else:
            results = all_results[:3]
            title_suffix = " (Top-3)"
        
        # Görüntüyü oku
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Görselleştir
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Görüntü
        axes[0].imshow(img)
        axes[0].axis('off')
        axes[0].set_title('Giris Goruntusu', fontsize=14, fontweight='bold')
        
        # Tahmin sonuçları
        classes = [r[0] for r in results]
        probs = [r[1] for r in results]
        
        # Renkleri ayarla (yüksek olasılık yeşil, düşük kırmızı)
        colors = plt.cm.RdYlGn(np.array(probs))
        axes[1].barh(classes, probs, color=colors)
        axes[1].set_xlabel('Olasilik', fontsize=12)
        axes[1].set_title(f'Tahmin Sonuclari{title_suffix}', fontsize=14, fontweight='bold')
        axes[1].set_xlim(0, 1)
        
        # En yüksek tahmin
        top_class, top_prob = all_results[0]
        fig.suptitle(
            f'Tahmin: {top_class} ({top_prob*100:.2f}%)',
            fontsize=16,
            fontweight='bold'
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Gorsellestirme kaydedildi: {save_path}")
        
        plt.show()
        
        # Konsola tüm sonuçları yazdır
        print(f"\n[INFO] Tum Tahmin Sonuclari (Siralanmis):")
        for i, (class_name, prob) in enumerate(all_results, 1):
            marker = ">>>" if i == 1 else "   "
            print(f"  {marker} {i}. {class_name}: {prob*100:.2f}%")
    
    def evaluate_on_test_set(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: Optional[List[str]] = None
    ):
        """
        Test seti üzerinde detaylı değerlendirme
        
        Args:
            X_test: Test görüntüleri
            y_test: Test etiketleri
            class_names: Sınıf isimleri
        """
        if class_names is None:
            class_names = self.class_names
        
        print("\n" + "=" * 60)
        print("DETAYLI DEĞERLENDİRME")
        print("=" * 60)
        
        # Tahminler
        y_pred = self.model.predict(X_test, verbose=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Classification report
        print("\n[INFO] Classification Report:")
        print(classification_report(
            y_true_classes,
            y_pred_classes,
            target_names=class_names
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Görüntü Sayısı'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Tahmin Edilen', fontsize=12)
        plt.ylabel('Gerçek', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        cm_path = 'results/confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"\n[OK] Confusion matrix kaydedildi: {cm_path}")
        plt.show()
        
        # Doğruluk hesapla
        accuracy = np.sum(y_pred_classes == y_true_classes) / len(y_true_classes)
        print(f"\n[OK] Toplam Dogruluk: {accuracy*100:.2f}%")
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'true_labels': y_true_classes,
            'predicted_labels': y_pred_classes
        }

def main():
    """Örnek kullanım"""
    # Model yolu (eğitim sonrası oluşturulan)
    model_path = 'models/best_model_transfer_efficientnet_*.h5'
    
    # En son modeli bul
    import glob
    models = glob.glob(model_path)
    if not models:
        print("[UYARI] Model bulunamadi! Lutfen once egitim yapin.")
        return
    
    latest_model = max(models, key=os.path.getctime)
    print(f"Kullanılan model: {latest_model}")
    
    # Predictor oluştur
    predictor = SkinDiseasePredictor(latest_model)
    
    # Örnek tahmin (test görüntüsü yolu gerekli)
    # predictor.visualize_prediction('path/to/test/image.jpg')

if __name__ == "__main__":
    main()

