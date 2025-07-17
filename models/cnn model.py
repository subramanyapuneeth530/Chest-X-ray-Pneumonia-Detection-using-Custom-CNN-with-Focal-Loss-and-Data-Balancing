import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import Callback

# --- 🔁 MODIFY THIS PATH ---
BASE_DIR = r"C:\Users\Puneeth\Downloads\archive\chest_xray"

# --- Image generators ---
train_datagen = ImageDataGenerator(rescale=1./255)
test_val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'train'),
    target_size=(150, 150),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

val_gen = test_val_datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'val'),
    target_size=(150, 150),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

test_gen = test_val_datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'test'),
    target_size=(150, 150),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# --- Custom Callback for Metrics ---
class MetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        val_preds = (self.model.predict(val_gen) > 0.5).astype("int32")
        val_labels = val_gen.classes
        precision = precision_score(val_labels, val_preds)
        recall = recall_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds)

        print(f"\nEpoch {epoch+1}: Accuracy = {logs['val_accuracy']:.4f}, "
              f"Loss = {logs['val_loss']:.4f}, "
              f"Precision = {precision:.4f}, Recall = {recall:.4f}, F1 Score = {f1:.4f}")

# --- Model ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)),
    MaxPooling2D(2, 2),
    BatchNormalization(),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    BatchNormalization(),

    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# --- Train ---
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    callbacks=[MetricsCallback()],
    verbose=0
)

# --- Plot training accuracy ---
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# --- Confusion Matrix Function ---
def plot_conf_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

# --- Train Confusion Matrix ---
train_preds = (model.predict(train_gen) > 0.5).astype("int32")
plot_conf_matrix(train_gen.classes, train_preds, "Train Confusion Matrix")

# --- Test Confusion Matrix ---
test_preds = (model.predict(test_gen) > 0.5).astype("int32")
plot_conf_matrix(test_gen.classes, test_preds, "Test Confusion Matrix")


#result
#| Metric    | **Train Set** | **Test Set** |
# | --------- | ------------- | ------------ |
# | Accuracy  | \~58.6%       | **82.8%** ✅  |
# | Precision | \~75.2%       | **85.3%** ✅  |
# | Recall    | \~66.1%       | **87.7%** ✅  |
# | F1 Score  | \~70.4%       | **86.5%** ✅  |
