"""
Model eğitim scripti
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
    TensorBoard
)
from typing import Tuple, Optional
import json
from datetime import datetime

# Windows konsolu için encoding ayarla
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

try:
    from .data_loader import SkinDiseaseDataLoader
    from .model import SkinDiseaseClassifier, create_simple_cnn
except ImportError:
    from data_loader import SkinDiseaseDataLoader
    from model import SkinDiseaseClassifier, create_simple_cnn

class ModelTrainer:
    """Model eğitimi için sınıf"""
    
    def __init__(
        self,
        model_type: str = 'transfer',
        base_model: str = 'efficientnet',
        img_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        epochs: int = 50
    ):
        """
        Args:
            model_type: 'transfer' veya 'simple'
            base_model: Transfer learning için ('efficientnet', 'resnet', 'mobilenet')
            img_size: Görüntü boyutu
            batch_size: Batch boyutu
            epochs: Epoch sayısı
        """
        self.model_type = model_type
        self.base_model = base_model
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.data_loader = SkinDiseaseDataLoader(img_size=img_size)
        self.model = None
        self.history = None
        self.class_weights = None  # Class weights için
        
        # Klasörleri oluştur
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('results', exist_ok=True)
    
    def load_data(self):
        """Veriyi yükle ve böl"""
        print("=" * 60)
        print("VERİ YÜKLEME")
        print("=" * 60)
        
        images, labels = self.data_loader.load_images()
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.split_data(
            images, labels
        )
        
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        # Class weights hesapla (Notebook'ta yok, bu yüzden kapalı)
        # Eğer çok dengesizlik varsa use_balanced=True yapabilirsiniz
        self.class_weights = self.calculate_class_weights(y_train, use_balanced=False)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def calculate_class_weights(self, y_train, use_balanced: bool = False):
        """Sınıf ağırlıklarını hesapla (dengesiz veri için)"""
        from sklearn.utils.class_weight import compute_class_weight
        
        # Her sınıfın örnek sayısını hesapla
        class_counts = np.sum(y_train, axis=0)
        total_samples = len(y_train)
        
        print("\n[INFO] Sinif Dagilimi:")
        for i, (class_name, count) in enumerate(zip(self.data_loader.SELECTED_CLASSES, class_counts)):
            percentage = (count / total_samples) * 100
            print(f"  {i+1}. {class_name}: {int(count)} goruntu ({percentage:.1f}%)")
        
        if not use_balanced:
            print("\n[INFO] Class weights KAPALI (Notebook'taki gibi)")
            print("[INFO] Model tum siniflari dengeli ogrenecek")
            return None
        
        # Class weights hesapla (sadece çok dengesiz durumlarda)
        # Ama çok agresif olmaması için sqrt kullan
        class_weights = compute_class_weight(
            'balanced',
            classes=np.arange(len(self.data_loader.SELECTED_CLASSES)),
            y=np.argmax(y_train, axis=1)
        )
        
        # Ağırlıkları yumuşat (sqrt ile)
        class_weights = np.sqrt(class_weights)
        class_weights = class_weights / class_weights.mean()  # Normalize et
        
        # Dictionary formatına çevir
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        print("\n[INFO] Class Weights (Yumusatilmis):")
        for i, (class_name, weight) in enumerate(zip(self.data_loader.SELECTED_CLASSES, class_weights)):
            print(f"  {class_name}: {weight:.2f}")
        
        return class_weight_dict
    
    def build_model(self, num_classes: int = 10):
        """Modeli oluştur"""
        print("\n" + "=" * 60)
        print("MODEL OLUŞTURMA")
        print("=" * 60)
        
        input_shape = (*self.img_size, 3)
        
        if self.model_type == 'transfer':
            classifier = SkinDiseaseClassifier(
                input_shape=input_shape,
                num_classes=num_classes,
                base_model_name=self.base_model
            )
            self.model = classifier.build_model(freeze_base=True)
            # Notebook'taki başarılı yaklaşımı kullan
            classifier.compile_model(learning_rate=0.001)
            # Adamax'a çevir (Notebook'ta başarılı)
            classifier.model.compile(
                optimizer=keras.optimizers.Adamax(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3)]
            )
            self.model = classifier.model
            
        elif self.model_type == 'simple':
            self.model = create_simple_cnn(input_shape, num_classes)
            # Notebook'taki gibi Adamax optimizer kullan
            self.model.compile(
                optimizer=keras.optimizers.Adamax(learning_rate=0.001),  # Adamax (Notebook'taki gibi)
                loss='categorical_crossentropy',
                metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3)]
            )
        else:
            raise ValueError(f"Bilinmeyen model tipi: {self.model_type}")
        
        print("\n[INFO] Model Ozeti:")
        self.model.summary()
        
        return self.model
    
    def setup_callbacks(self):
        """Callback'leri ayarla"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{self.model_type}_{self.base_model}_{timestamp}"
        
        callbacks = []
        
        # Model checkpoint
        checkpoint = ModelCheckpoint(
            f'models/best_model_{model_name}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping - Notebook'ta yok, ama overfitting'i önlemek için var
        # Daha esnek yapıyoruz (patience artırıldı)
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=25,  # Artırıldı (önceden 15'ti)
            restore_best_weights=True,
            verbose=1,
            min_delta=0.0001  # Minimum iyileşme
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction - Daha yavaş azaltma
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,  # Daha yavaş azaltma (önceden 0.5'ti)
            patience=8,  # Daha sabırlı (önceden 5'ti)
            min_lr=1e-6,  # Daha yüksek minimum LR
            verbose=1,
            cooldown=2  # Cooldown ekle
        )
        callbacks.append(reduce_lr)
        
        # CSV logger
        csv_logger = CSVLogger(
            f'logs/training_{model_name}.csv',
            append=False
        )
        callbacks.append(csv_logger)
        
        # TensorBoard (opsiyonel)
        tensorboard = TensorBoard(
            log_dir=f'logs/tensorboard_{model_name}',
            histogram_freq=1
        )
        callbacks.append(tensorboard)
        
        self.model_name = model_name
        return callbacks
    
    def train(self):
        """Modeli eğit"""
        print("\n" + "=" * 60)
        print("MODEL EĞİTİMİ")
        print("=" * 60)
        
        callbacks = self.setup_callbacks()
        
        print(f"\n⚙️  Eğitim Parametreleri:")
        print(f"  Model tipi: {self.model_type}")
        print(f"  Base model: {self.base_model}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Görüntü boyutu: {self.img_size}")
        
        print(f"\n[INFO] Egitim basliyor...\n")
        
        # Class weights kullan (sınıf dengesizliği için)
        fit_params = {
            'x': self.X_train,
            'y': self.y_train,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'validation_data': (self.X_val, self.y_val),
            'callbacks': callbacks,
            'verbose': 1
        }
        
        if self.class_weights is not None:
            fit_params['class_weight'] = self.class_weights
            print("[INFO] Class weights aktif (sinif dengesizligi duzeltiliyor)")
        
        self.history = self.model.fit(**fit_params)
        
        print("\n[OK] Egitim tamamlandi!")
        return self.history
    
    def evaluate(self):
        """Modeli değerlendir"""
        print("\n" + "=" * 60)
        print("MODEL DEĞERLENDİRME")
        print("=" * 60)
        
        # Test seti üzerinde değerlendir
        test_results = self.model.evaluate(
            self.X_test,
            self.y_test,
            verbose=1
        )
        
        print(f"\n[INFO] Test Seti Sonuclari:")
        print(f"  Loss: {test_results[0]:.4f}")
        print(f"  Accuracy: {test_results[1]:.4f}")
        if len(test_results) > 2:
            print(f"  Top-3 Accuracy: {test_results[2]:.4f}")
        
        return test_results
    
    def plot_history(self):
        """Eğitim geçmişini görselleştir"""
        if self.history is None:
            print("Eğitim geçmişi bulunamadı!")
            return
        
        history = self.history.history
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        axes[0].plot(history['accuracy'], label='Eğitim', linewidth=2)
        axes[0].plot(history['val_accuracy'], label='Doğrulama', linewidth=2)
        axes[0].set_title('Model Doğruluğu', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(history['loss'], label='Eğitim', linewidth=2)
        axes[1].plot(history['val_loss'], label='Doğrulama', linewidth=2)
        axes[1].set_title('Model Kaybı', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Kaydet
        plot_path = f'results/training_history_{self.model_name}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Grafik kaydedildi: {plot_path}")
        
        plt.show()
    
    def save_results(self):
        """Sonuçları kaydet"""
        if self.history is None:
            return
        
        history_dict = {k: [float(v) for v in values] 
                       for k, values in self.history.history.items()}
        
        results = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'base_model': self.base_model,
            'img_size': self.img_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'final_train_accuracy': float(self.history.history['accuracy'][-1]),
            'final_val_accuracy': float(self.history.history['val_accuracy'][-1]),
            'best_val_accuracy': float(max(self.history.history['val_accuracy'])),
            'history': history_dict
        }
        
        results_path = f'results/results_{self.model_name}.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Sonuclar kaydedildi: {results_path}")

def main():
    """Ana fonksiyon"""
    # Eğitici oluştur
    trainer = ModelTrainer(
        model_type='transfer',  # 'transfer' veya 'simple'
        base_model='efficientnet',  # 'efficientnet', 'resnet', 'mobilenet'
        img_size=(224, 224),
        batch_size=32,
        epochs=50
    )
    
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

if __name__ == "__main__":
    main()

