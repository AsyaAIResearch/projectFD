import tensorflow as tf
import os
import random
import numpy as np
import pandas as pd
from keras.layers import BatchNormalization
from keras_preprocessing.image import ImageDataGenerator
from keras_tuner import Hyperband
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Input, Concatenate, UpSampling2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

base_dir = r'/content/drive/MyDrive/MulticlassDataset'
os.makedirs(base_dir, exist_ok=True)  # Eğer dizin yoksa oluştur

train_dir = r'/content/drive/MyDrive/MulticlassDataset/train/normal'
val_dir = r'/content/drive/MyDrive/MulticlassDataset/val/normal'
test_dir = r'/content/drive/MyDrive/MulticlassDataset/test/normal'

img_width, img_height = 224, 224
batch_size = 32
epochs = 50
learning_rate = 0.001
num_classes = 7
dropout_rate = 0.5

filter_sizes = [32, 64, 128]  # Farklı katmanlar için filtre sayıları
kernel_sizes = [(3, 3), (5, 5)]  # Konvolüsyon katmanlarındaki çekirdek boyutları
dropout_rates = [0.3, 0.5, 0.7]
learning_rates = [0.0001, 0.00001, 0.001]

# Veri artırma
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'  # Çok sınıflı sınıflandırma için 'categorical' kullanılıyor
)

# Doğrulama veri seti oluşturuluyor
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Test veri seti oluşturuluyor
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Modeli oluşturan fonksiyon
def build_model(input_shape=(224, 224, 3), classes=7, kernel_size=3, filter_depth=(64, 128, 256)):
    img_input = Input(shape=input_shape)  # Giriş katmanı, resim boyutları belirlenir

    # Encoder Bölümü (Özelliklerin çıkarıldığı kısım)
    conv1 = Conv2D(filter_depth[0], (kernel_size, kernel_size), padding="same")(img_input)
    batch1 = BatchNormalization()(conv1)
    act1 = Activation("relu")(batch1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(act1)

    conv2 = Conv2D(filter_depth[1], (kernel_size, kernel_size), padding="same")(pool1)
    batch2 = BatchNormalization()(conv2)
    act2 = Activation("relu")(batch2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(act2)

    conv3 = Conv2D(filter_depth[2], (kernel_size, kernel_size), padding="same")(pool2)
    batch3 = BatchNormalization()(conv3)
    act3 = Activation("relu")(batch3)

    # Decoder Bölümü (Görüntünün tekrar büyütüldüğü kısım)
    up1 = UpSampling2D(size=(2, 2))(act3)  # Görüntü boyutunu büyütme işlemi
    conv4 = Conv2D(filter_depth[1], (kernel_size, kernel_size), padding="same")(up1)
    batch4 = BatchNormalization()(conv4)
    act4 = Activation("relu")(batch4)

    up2 = UpSampling2D(size=(2, 2))(act4)
    conv5 = Conv2D(filter_depth[0], (kernel_size, kernel_size), padding="same")(up2)
    batch5 = BatchNormalization()(conv5)
    act5 = Activation("relu")(batch5)

    # Çıkış katmanı
    conv6 = Conv2D(classes, (1, 1), activation="softmax", padding="same")(act5)  # Softmax ile sınıflandırma yapılır
    output = tf.keras.layers.GlobalAveragePooling2D()(conv6)  # Boyutları doğrudan küçültme
    model = Model(img_input, output)

    return model

# Belirli sayıda deneme yaparak en iyi modeli ve hiperparametreleri bulur
def random_search(num_trials=5):
    best_model = None  # En iyi modeli saklamak için değişken
    best_acc = 0  # En iyi doğruluk oranını saklamak için değişken
    best_params = {}  # En iyi parametreleri saklamak için sözlük

    for trial in range(num_trials):
        # Her deneme için rastgele hiperparametre seçimi
        filters = random.choice(filter_sizes)
        kernel_size = random.choice(kernel_sizes)
        dropout_rate = random.choice(dropout_rates)
        lr = random.choice(learning_rates)

        print(f"Trial {trial+1}: filters={filters}, kernel_size={kernel_size}, dropout_rate={dropout_rate}, learning_rate={lr}")

        model = build_model(input_shape=(img_width, img_height, 3), classes=num_classes)

        # Optimizasyon fonksiyonunu belirlenen öğrenme oranı ile güncelle
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Modeli eğit ve doğrulama seti üzerinde değerlendir
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[early_stopping, model_checkpoint]
        )

        # En iyi doğruluk oranını güncelle
        val_acc = max(history.history['val_accuracy'])
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model
            best_params = {
                'filters': filters,
                'kernel_size': kernel_size,
                'dropout_rate': dropout_rate,
                'learning_rate': lr
            }

    return best_model, best_params, best_acc

checkpoint_path = os.path.join(base_dir, "best_model2.keras")
model_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

best_model, best_params, best_acc = random_search()

print("Best Model Parameters:", best_params)
print(f"Best Validation Accuracy: {best_acc}")

history_path = os.path.join(base_dir, "training_history2.npy")
np.save(history_path, best_model.history.history)

# Test veri kümesi üzerinde tahmin yap
y_true = test_generator.classes  # Gerçek etiketler
y_pred_probs = best_model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)  # En yüksek olasılıklı sınıfı seç

predictions_path = os.path.join(base_dir, "predictions2.csv")
pred_df = pd.DataFrame({
    "True Label": y_true,
    "Predicted Label": y_pred
})
pred_df.to_csv(predictions_path, index=False)

class_labels = list(test_generator.class_indices.keys())
report = classification_report(y_true, y_pred, target_names=class_labels)
report_path = os.path.join(base_dir, "classification_report2.txt")
with open(report_path, "w") as f:
    f.write(report)

print("Classification Report:")
print(report)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt_path = os.path.join(base_dir, "confusion_matrix2.png")
plt.savefig(plt_path)
plt.show()

print(best_model.history.history.keys())

plt.tight_layout()
results_plot_path = os.path.join(base_dir, "training_results2.png")
plt.savefig(results_plot_path)
plt.show()

final_model_path = os.path.join(base_dir, "final_model2.keras")
best_model.save(final_model_path)
print(f"Model and results saved in: {base_dir}")