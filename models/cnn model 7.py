from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import tensorflow as tf

# Define a reusable convolutional block
def conv_block(filters):
    return [
        Conv2D(filters, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(filters, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3)
    ]
# Paths
base_path = "C:/Users/Puneeth/Downloads/archive/chest_xray"
train_dir = os.path.join(base_path, "balanced_train")
val_dir = os.path.join(base_path, "val")
test_dir = os.path.join(base_path, "test")

# Parameters
img_size = (256, 256)
batch_size = 32
epochs = 25
learning_rate = 1e-4

# Data Generators
train_gen = ImageDataGenerator(rescale=1./255, rotation_range=15, zoom_range=0.1, horizontal_flip=True)
val_test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
val_data = val_test_gen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
test_data = val_test_gen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False)
model = Sequential([
    Input(shape=(img_size[0], img_size[1], 3)),

    *conv_block(32),
    *conv_block(64),
    *conv_block(128),

    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
    tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, verbose=1)
]

history = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=callbacks)
# Predictions
y_true = test_data.classes
y_pred_prob = model.predict(test_data).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

# Scores
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = model.evaluate(test_data, verbose=0)[1]

print(f"\nTest Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
plt.title('Test Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Training Curves
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over Epochs')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()

# Test Accuracy: 81.09%
# Precision: 0.84
# Recall: 0.86
# F1 Score: 0.85