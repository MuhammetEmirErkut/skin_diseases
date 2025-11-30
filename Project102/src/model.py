"""
CNN Model tanımlamaları - Transfer Learning ile geliştirilmiş modeller
"""
import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0, ResNet50, MobileNetV2
from tensorflow.keras.optimizers import Adam
from typing import Tuple, Optional

# Windows konsolu için encoding ayarla
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

class SkinDiseaseClassifier:
    """Cilt hastalığı sınıflandırıcı model sınıfı"""
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 10,
        base_model_name: str = 'efficientnet'
    ):
        """
        Args:
            input_shape: Giriş görüntü boyutu
            num_classes: Sınıf sayısı
            base_model_name: Kullanılacak base model ('efficientnet', 'resnet', 'mobilenet')
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_model_name = base_model_name.lower()
        self.model = None
    
    def _get_base_model(self):
        """Base modeli seç ve yükle"""
        if self.base_model_name == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            print("[OK] EfficientNetB0 base model yuklendi")
            
        elif self.base_model_name == 'resnet':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            print("[OK] ResNet50 base model yuklendi")
            
        elif self.base_model_name == 'mobilenet':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            print("[OK] MobileNetV2 base model yuklendi")
            
        else:
            raise ValueError(
                f"Bilinmeyen base model: {self.base_model_name}. "
                "Desteklenenler: 'efficientnet', 'resnet', 'mobilenet'"
            )
        
        return base_model
    
    def build_model(self, freeze_base: bool = True, dropout_rate: float = 0.5):
        """
        Transfer Learning ile model oluştur
        
        Args:
            freeze_base: Base model katmanlarını dondur (fine-tuning için False yapın)
            dropout_rate: Dropout oranı
        """
        # Base modeli yükle
        base_model = self._get_base_model()
        
        # Base model katmanlarını dondur (opsiyonel)
        if freeze_base:
            base_model.trainable = False
            print("[OK] Base model katmanlari donduruldu")
        else:
            # Son birkaç katmanı fine-tuning için aç
            base_model.trainable = True
            for layer in base_model.layers[:-20]:
                layer.trainable = False
            print("[OK] Base model katmanlari fine-tuning icin hazirlandi")
        
        # Model mimarisi
        inputs = layers.Input(shape=self.input_shape)
        
        # Data augmentation (eğitim sırasında)
        x = layers.RandomRotation(0.1)(inputs)
        x = layers.RandomTranslation(0.1, 0.1)(x)
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomZoom(0.1)(x)
        
        # Base model
        x = base_model(x, training=False)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Batch Normalization
        x = layers.BatchNormalization()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate * 0.5)(x)
        
        # Çıkış katmanı
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Modeli oluştur
        self.model = Model(inputs=inputs, outputs=outputs)
        
        return self.model
    
    def compile_model(
        self,
        learning_rate: float = 0.001,
        optimizer: Optional[keras.optimizers.Optimizer] = None,
        use_adamax: bool = True  # Notebook'taki gibi Adamax kullan
    ):
        """Modeli derle"""
        if self.model is None:
            raise ValueError("Önce build_model() çağrılmalı!")
        
        if optimizer is None:
            if use_adamax:
                optimizer = keras.optimizers.Adamax(learning_rate=learning_rate)
                print("[INFO] Adamax optimizer kullaniliyor (Notebook'taki gibi)")
            else:
                optimizer = Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
            ]
        )
        
        print("[OK] Model derlendi")
        return self.model
    
    def get_model_summary(self):
        """Model özetini yazdır"""
        if self.model is None:
            print("Model henüz oluşturulmadı!")
            return
        
        self.model.summary()
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        
        print(f"\n[INFO] Model Istatistikleri:")
        print(f"  Toplam parametre: {total_params:,}")
        print(f"  Eğitilebilir parametre: {trainable_params:,}")
        print(f"  Dondurulmuş parametre: {total_params - trainable_params:,}")

def create_simple_cnn(input_shape: Tuple[int, int, int], num_classes: int):
    """
    Basit CNN modeli (Transfer Learning olmadan)
    Notebook'taki modelden esinlenilmiştir ama geliştirilmiştir
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # Data Augmentation
        layers.RandomRotation(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(0.1),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

