import pathlib
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from ultralytics import YOLO

dataset_path = pathlib.Path("Kırık Görüntüleri")
label_path = pathlib.Path("Etiket Görüntüleri")

def get_sorted_files(image_dir, label_dir):
    image_files = sorted(image_dir.glob('*.jpg'))  # JPG formatındaki tüm resimleri al
    label_files = sorted(label_dir.glob('*.txt'))  # TXT formatındaki tüm etiket dosyalarını al
    return image_files, label_files

def get_label(file_path):
    with open(file_path, 'r') as file:
        label_content = file.readline().strip()  # İlk satırı oku ve boşlukları temizle
        if not label_content:
            return None
        return int(label_content[0])

# Resim ve etiket dosyalarını al
image_files, label_files = get_sorted_files(dataset_path, label_path)

images = []
labels = []
IMG_SIZE = 256  # yeniden boyutlandırma

for image_file, label_file in zip(image_files, label_files):
    image = cv2.imread(str(image_file))
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    label = get_label(str(label_file))
    if label is None:
        continue  # Etiket yoksa bu veriyi atla
    images.append(image)  # Görüntüyü listeye ekle
    labels.append(label)  # Etiketi listeye ekle

# Listeleri numpy array'e çevir
images = np.array(images)
labels = np.array(labels)

# Etiketleri belirlenen değerlere göre dönüştür
label_mapping = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5}  # Yeni etiketleme düzeni
labels_mapped = np.array([label_mapping[label] for label in labels])  # Etiketleri dönüştür

# Kategorik hale getir
labels_one_hot = to_categorical(labels_mapped, num_classes=6)

# Veriyi eğitim, doğrulama ve test setlerine ayır
X_train, X_temp, y_train, y_temp = train_test_split(images, labels_mapped, test_size=0.3, stratify=labels_mapped)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)

# Görselleri normalleştir
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

# Veri artırma
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,  # genişlik kaydırma
    height_shift_range=0.2,  # yükseklik kaydırma
    shear_range=0.2,  # kırpma
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'  # yeni pikselleri en yakın değerlere göre doldurma
)

# Eğitim verisi için artırma işlemini uygula
train_data = datagen.flow(X_train, to_categorical(y_train, num_classes=6), batch_size=32, shuffle=True)

# Doğrulama verisi için yalnızca ölçekleme uygula
validation_data = ImageDataGenerator(rescale=1./255).flow(
    X_val, to_categorical(y_val, num_classes=6), batch_size=32
)

# Modelin oluşturulması
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),  # İlk evrişim katmanı
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),  # İkinci evrişim katmanı
    MaxPooling2D(),
    Conv2D(128, (3, 3), activation='relu'),  # Üçüncü evrişim katmanı
    MaxPooling2D(),
    Flatten(),
    Dropout(0.3),  # %30 oranında dropout eklenerek aşırı öğrenmeyi önleme
    Dense(128, activation='relu'),  # Tam bağlı katman
    Dense(6, activation='softmax')  # Çıkış katmanı
])

# Modelin yapısını ekrana yazdır
model.summary()

# Modelin derlenmesi
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',  # Kayıp fonksiyonu
    metrics=['accuracy']
)

# ModelCheckpoint ile en iyi modeli kaydet
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# EarlyStopping ile modelin fazla eğitimi engellendi
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Modelin eğitilmesi
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=50,
    callbacks=[model_checkpoint, early_stopping]
)

# Modelin test verisi üzerinde değerlendirilmesi
test_loss, test_accuracy = model.evaluate(X_test, to_categorical(y_test, num_classes=6))
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

plt.figure(figsize=(12, 6))

# Doğruluk (accuracy) grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Kayıp (loss) grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Modelin test verileri üzerindeki tahminleri
y_pred = np.argmax(model.predict(X_test), axis=1)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

print("Classification Report:\n", classification_report(y_test, y_pred))

example_image = X_test[0]
example_label = y_test[0]
predicted_label = np.argmax(model.predict(example_image.reshape(1, IMG_SIZE, IMG_SIZE, 3)), axis=1)

plt.imshow(example_image)
plt.title(f"True Label: {example_label}, Predicted: {predicted_label[0]}")
plt.show()

# YOLO modelinin yüklenmesi
model = YOLO("yolov8s.pt")

# YOLO modelinin eğitilmesi
model.train(
    data=r'C:\Users\kuzey\PycharmProjects\tumkodlarFD\data.yaml',  # Veri kümesi yolu
    epochs=20,
    imgsz=640,
    batch=16,
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,  # Ağırlık çürümesi
    workers=16,
    project='yolo_project_sobel',
    name='yolo_experiment_sobel',
    save=True,
    save_period=1,
    patience=0,
)