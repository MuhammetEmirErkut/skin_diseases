"""
Geliştirilmiş Model Eğitim Sistemi
Notebook'taki başarılı yaklaşımı kullanarak yüksek doğruluk sağlar
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
    TensorBoard,
    LearningRateScheduler
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

class ImprovedModelTrainer:
    """Geliştirilmiş model eğitici - Notebook'taki başarılı yaklaşımı kullanır"""
    
    def __init__(
        self,
        model_type: str = 'transfer',
        base_model: str = 'efficientnet',
        img_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        epochs: int = 50,
        use_class_weights: bool = False  # Notebook'ta yok, kapalı
    ):
        """
        Args:
            model_type: 'transfer' veya 'simple'
            base_model: Transfer learning için ('efficientnet', 'resnet', 'mobilenet')
            img_size: Görüntü boyutu
            batch_size: Batch boyutu
            epochs: Epoch sayısı
            use_class_weights: Class weights kullan (Notebook'ta yok, False önerilir)
        """
        self.model_type = model_type
        self.base_model = base_model
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_class_weights = use_class_weights
        
        self.data_loader = SkinDiseaseDataLoader(img_size=img_size)
        self.model = None
        self.history = None
        self.class_weights = None
        
        # Klasörleri oluştur
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('results', exist_ok=True)
    
    def load_data(self):
        """Veriyi yükle ve böl"""
        print("=" * 70)
        print("VERİ YÜKLEME")
        print("=" * 70)
        
        images, labels = self.data_loader.load_images()
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.split_data(
            images, labels,
            test_size=0.15,  # Notebook'taki gibi
            val_size=0.15
        )
        
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        # Class weights (opsiyonel, Notebook'ta yok)
        if self.use_class_weights:
            self.class_weights = self.calculate_class_weights(y_train)
        else:
            print("\n[INFO] Class weights KAPALI (Notebook'taki gibi)")
            print("[INFO] Model tum siniflari dengeli ogrenecek")
            self.class_weights = None
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def calculate_class_weights(self, y_train):
        """Sınıf ağırlıklarını hesapla (sadece gerekirse)"""
        from sklearn.utils.class_weight import compute_class_weight
        
        class_counts = np.sum(y_train, axis=0)
        total_samples = len(y_train)
        
        print("\n[INFO] Sinif Dagilimi:")
        for i, (class_name, count) in enumerate(zip(self.data_loader.SELECTED_CLASSES, class_counts)):
            percentage = (count / total_samples) * 100
            print(f"  {i+1}. {class_name}: {int(count)} goruntu ({percentage:.1f}%)")
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.arange(len(self.data_loader.SELECTED_CLASSES)),
            y=np.argmax(y_train, axis=1)
        )
        
        # Yumuşatılmış ağırlıklar
        class_weights = np.sqrt(class_weights)
        class_weights = class_weights / class_weights.mean()
        
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        print("\n[INFO] Class Weights (Yumusatilmis):")
        for i, (class_name, weight) in enumerate(zip(self.data_loader.SELECTED_CLASSES, class_weights)):
            print(f"  {class_name}: {weight:.2f}")
        
        return class_weight_dict
    
    def build_model(self, num_classes: int = 10):
        """Modeli oluştur - Notebook'taki başarılı yaklaşımı kullan"""
        print("\n" + "=" * 70)
        print("MODEL OLUŞTURMA (Notebook Yaklaşımı)")
        print("=" * 70)
        
        input_shape = (*self.img_size, 3)
        
        if self.model_type == 'transfer':
            classifier = SkinDiseaseClassifier(
                input_shape=input_shape,
                num_classes=num_classes,
                base_model_name=self.base_model
            )
            self.model = classifier.build_model(freeze_base=True)
            # Notebook'taki gibi Adamax kullan
            classifier.compile_model(learning_rate=0.001, use_adamax=True)
            self.model = classifier.model
            
        elif self.model_type == 'simple':
            # Notebook'taki basit CNN modeli
            self.model = create_simple_cnn(input_shape, num_classes)
            # Notebook'taki gibi Adamax optimizer
            self.model.compile(
                optimizer=keras.optimizers.Adamax(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3)]
            )
        else:
            raise ValueError(f"Bilinmeyen model tipi: {self.model_type}")
        
        print("\n[INFO] Model Ozeti:")
        self.model.summary()
        
        return self.model
    
    def setup_callbacks(self):
        """Callback'leri ayarla - Notebook yaklaşımına uygun"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{self.model_type}_{self.base_model}_{timestamp}"
        
        callbacks = []
        
        # Model checkpoint - En iyi modeli kaydet
        checkpoint = ModelCheckpoint(
            f'models/best_model_{model_name}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping - Çok esnek (Notebook'ta yok ama overfitting için)
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=30,  # Çok sabırlı (Notebook'ta 20 epoch var)
            restore_best_weights=True,
            verbose=1,
            min_delta=0.0001
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction - Yavaş ve sabırlı
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,  # Yavaş azaltma
            patience=10,  # Sabırlı
            min_lr=1e-6,
            verbose=1,
            cooldown=3
        )
        callbacks.append(reduce_lr)
        
        # CSV logger
        csv_logger = CSVLogger(
            f'logs/training_{model_name}.csv',
            append=False
        )
        callbacks.append(csv_logger)
        
        self.model_name = model_name
        return callbacks
    
    def train(self):
        """Modeli eğit - Notebook'taki başarılı yaklaşımı kullan"""
        print("\n" + "=" * 70)
        print("MODEL EĞİTİMİ (Notebook Yaklaşımı)")
        print("=" * 70)
        
        callbacks = self.setup_callbacks()
        
        print(f"\n[INFO] Egitim Parametreleri:")
        print(f"  Model tipi: {self.model_type}")
        print(f"  Base model: {self.base_model}")
        print(f"  Optimizer: Adamax (Notebook'taki gibi)")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Goruntu boyutu: {self.img_size}")
        print(f"  Class weights: {'AKTIF' if self.class_weights else 'KAPALI (Notebook gibi)'}")
        
        print(f"\n[INFO] Egitim basliyor...\n")
        
        # Eğitim parametreleri
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
        
        self.history = self.model.fit(**fit_params)
        
        print("\n[OK] Egitim tamamlandi!")
        return self.history
    
    def evaluate(self):
        """Modeli değerlendir"""
        print("\n" + "=" * 70)
        print("MODEL DEĞERLENDİRME")
        print("=" * 70)
        
        test_results = self.model.evaluate(
            self.X_test,
            self.y_test,
            verbose=1
        )
        
        print(f"\n[INFO] Test Seti Sonuclari:")
        print(f"  Loss: {test_results[0]:.4f}")
        print(f"  Accuracy: {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
        if len(test_results) > 2:
            print(f"  Top-3 Accuracy: {test_results[2]:.4f} ({test_results[2]*100:.2f}%)")
        
        # En iyi validation accuracy
        if self.history:
            best_val_acc = max(self.history.history['val_accuracy'])
            print(f"\n[INFO] En Iyi Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        
        return test_results
    
    def plot_history(self):
        """Eğitim geçmişini görselleştir"""
        if self.history is None:
            print("Egitim gecmisi bulunamadi!")
            return
        
        history = self.history.history
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        axes[0].plot(history['accuracy'], label='Egitim', linewidth=2)
        axes[0].plot(history['val_accuracy'], label='Dogrulama', linewidth=2)
        axes[0].set_title('Model Dogrulugu', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(history['loss'], label='Egitim', linewidth=2)
        axes[1].plot(history['val_loss'], label='Dogrulama', linewidth=2)
        axes[1].set_title('Model Kaybi', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
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
        
        best_val_acc = max(self.history.history['val_accuracy'])
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        
        results = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'base_model': self.base_model,
            'img_size': self.img_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'optimizer': 'Adamax',
            'use_class_weights': self.use_class_weights,
            'final_train_accuracy': float(final_train_acc),
            'final_val_accuracy': float(final_val_acc),
            'best_val_accuracy': float(best_val_acc),
            'history': history_dict
        }
        
        results_path = f'results/results_{self.model_name}.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Sonuclar kaydedildi: {results_path}")
        
        # Başarı özeti
        print("\n" + "=" * 70)
        print("BAŞARI ÖZETİ")
        print("=" * 70)
        print(f"  Final Train Accuracy: {final_train_acc*100:.2f}%")
        print(f"  Final Val Accuracy: {final_val_acc*100:.2f}%")
        print(f"  Best Val Accuracy: {best_val_acc*100:.2f}%")
        print("=" * 70)

def main():
    """Ana fonksiyon"""
    print("=" * 70)
    print(" " * 15 + "GELİŞTİRİLMİŞ MODEL EĞİTİMİ")
    print(" " * 10 + "Notebook Yaklaşımı ile Yüksek Doğruluk")
    print("=" * 70)
    
    # Eğitici oluştur
    trainer = ImprovedModelTrainer(
        model_type='transfer',  # 'transfer' veya 'simple'
        base_model='efficientnet',
        img_size=(224, 224),
        batch_size=32,
        epochs=50,
        use_class_weights=False  # Notebook'ta yok, False önerilir
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




