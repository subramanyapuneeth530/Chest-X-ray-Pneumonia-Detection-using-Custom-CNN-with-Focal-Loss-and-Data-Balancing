import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

# ======================
#  SETTINGS
# ======================
img_size = (256, 256)
batch_size = 32
epochs = 10

#  MODIFY THIS PATH
base_path = "C:/Users/Puneeth/Downloads/archive/chest_xray"
train_dir = os.path.join(base_path, "balanced_train")
val_dir = os.path.join(base_path, "val")
test_dir = os.path.join(base_path, "test")

# ======================
# DATA LOADERS
# ======================
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=True)
val_data = val_gen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False)
test_data = test_gen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False)

# ======================
#  CNN MODEL
# ======================
model = Sequential([
    Input(shape=(*img_size, 3)),

    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# ======================
#  TRAINING LOOP
# ======================
history = {
    'accuracy': [],
    'val_accuracy': [],
    'precision': [],
    'recall': [],
    'f1': []
}

for epoch in range(epochs):
    print(f"\n Epoch {epoch+1}/{epochs}")
    result = model.fit(train_data, validation_data=val_data, epochs=1, verbose=1)

    # Predictions on validation
    val_preds = model.predict(val_data, verbose=0) > 0.5
    val_true = val_data.classes

    precision = precision_score(val_true, val_preds)
    recall = recall_score(val_true, val_preds)
    f1 = f1_score(val_true, val_preds)

    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")

    history['accuracy'].append(result.history['accuracy'][0])
    history['val_accuracy'].append(result.history['val_accuracy'][0])
    history['precision'].append(precision)
    history['recall'].append(recall)
    history['f1'].append(f1)

# ======================
#  PLOT METRICS
# ======================
plt.plot(history['accuracy'], label='Train Acc')
plt.plot(history['val_accuracy'], label='Val Acc')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

# ======================
#  CONFUSION MATRICES
# ======================
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

# On Train Data
train_preds = model.predict(train_data, verbose=0) > 0.5
plot_confusion_matrix(train_data.classes, train_preds, "Train Confusion Matrix")

# On Test Data
test_preds = model.predict(test_data, verbose=0) > 0.5
plot_confusion_matrix(test_data.classes, test_preds, "Test Confusion Matrix")

# ======================
#  Final Test Report
# ======================
precision = precision_score(test_data.classes, test_preds)
recall = recall_score(test_data.classes, test_preds)
f1 = f1_score(test_data.classes, test_preds)
acc = np.mean(test_preds.flatten() == test_data.classes)

print("\n Final Test Performance:")
print(f"Accuracy  : {acc*100:.2f}%")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


#  Final Test Performance:
# Accuracy  : 66.35%
# Precision : 0.6500
# Recall    : 1.0000
# F1 Score  : 0.7879