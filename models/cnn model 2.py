import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------
# Set your dataset paths here
# ---------------------------
base_dir = "C:/Users/Puneeth/Downloads/archive/chest_xray"
train_dir = os.path.join(base_dir, "balanced_train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# ---------------------------
# Data Preprocessing
# ---------------------------
img_size = 224
batch_size = 32

train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=(img_size, img_size), color_mode='grayscale', class_mode='binary', batch_size=batch_size)
val_data = val_gen.flow_from_directory(val_dir, target_size=(img_size, img_size), color_mode='grayscale', class_mode='binary', batch_size=batch_size)
test_data = test_gen.flow_from_directory(test_dir, target_size=(img_size, img_size), color_mode='grayscale', class_mode='binary', batch_size=batch_size, shuffle=False)

# ---------------------------
# Model Architecture
# ---------------------------
model = Sequential([
    Input(shape=(img_size, img_size, 1)),

    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    BatchNormalization(),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    BatchNormalization(),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    BatchNormalization(),

    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ---------------------------
# Training
# ---------------------------
epochs = 10

history = model.fit(train_data, epochs=epochs, validation_data=val_data)

# ---------------------------
# Evaluate + Metrics
# ---------------------------
def evaluate_model(data, dataset_name="Test"):
    y_true = data.classes
    y_pred_prob = model.predict(data)
    y_pred = (y_pred_prob > 0.5).astype(int)

    acc = np.mean(y_true == y_pred.flatten())
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n {dataset_name} Results:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Pneumonia"], yticklabels=["Normal", "Pneumonia"])
    plt.title(f'{dataset_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# Evaluate on train and test
evaluate_model(train_data, "Train")
evaluate_model(test_data, "Test")

# ---------------------------
# Plot Accuracy vs Epoch
# ---------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Train Results:
# Accuracy:  0.4925
# Precision: 0.4923
# Recall:    0.4758
# F1 Score:  0.4839

#  Test Results:
# Accuracy:  0.8381
# Precision: 0.8247
# Recall:    0.9410
# F1 Score:  0.8790