# 🩺 Chest X-Ray Pneumonia Classification using CNN

This project focuses on building a custom Convolutional Neural Network (CNN) from scratch to classify chest X-ray images into two categories: Pneumonia and Normal. Using the publicly available chest X-ray dataset, the model was trained to recognize patterns in grayscale lung scans and achieved high performance metrics including over 91% test accuracy and an F1 score of 0.93 in its best configuration. Multiple model versions were tested to handle challenges like class imbalance and overfitting.

---

## 📁 Dataset Structure
The dataset consists of chest X-ray images organized into three directories — train, val, and test, each with two subdirectories: NORMAL and PNEUMONIA. The training data was initially imbalanced, containing more pneumonia samples than normal. To improve performance, a balanced dataset was later created by randomly undersampling the pneumonia images.
```bash
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## 🧠 Model Architecture 
The model is a custom Convolutional Neural Network (CNN) built using Keras' high-level Sequential API. It is specifically designed to classify grayscale chest X-ray images into two categories: Normal or Pneumonia. The convolutional backbone of this CNN consists of three sequential convolutional blocks, each designed to extract increasingly complex features from the chest X-ray images. These layers allow the model to interpret medical patterns ranging from basic edges to complex signs of pneumonia.

**🧱 Convolutional Layers**
* First Convolutional Block
  This block starts the feature extraction process using 32 filters of size 3×3 with ReLU activation. It learns simple and local patterns such as edges, textures, and corners. It is followed by a MaxPooling layer to reduce the spatial size and retain only the most significant features, and Batch Normalization to ensure stable and faster convergence.

* Second Convolutional Block
  The second block doubles the filters to 64, allowing the network to capture more refined patterns such as shapes, boundaries of lung fields, or denser regions in the scan. The MaxPooling and BatchNormalization operations are repeated to preserve essential information while continuing to regularize the model.

* Third Convolutional Block
  This final convolutional block uses 128 filters, enabling the model to detect complex, high-level medical features like lobar consolidations or interstitial opacities—key indicators in pneumonia diagnosis. These abstract features are critical for accurate classification between pneumonia and normal chest X-rays. Pooling and normalization finalize the block to enhance learning and reduce overfitting.

**🔄 Flatten and Fully Connected Layers**

After extracting spatial features through the convolutional blocks, the model uses a Flatten() layer to transform the 2D feature maps into a 1D vector. This flattening process essentially unrolls the learned visual information into a single list of numbers, preparing it for classification.

Following this, a Dense (fully connected) layer with 256 neurons and a ReLU activation function acts as the decision-making component of the network. It interprets the unrolled features to understand the presence or absence of pneumonia. To prevent over-reliance on specific patterns from the training data, a Dropout layer is applied, randomly deactivating 40% of the neurons during training. This helps the model learn more robust and generalized patterns rather than memorizing specific examples.

**✅ Output Layer**

The final layer in the network is a Dense layer with a single neuron and a sigmoid activation function, which outputs a probability score between 0 and 1. This score represents the likelihood of the input X-ray showing pneumonia. If the output is closer to 0, the model predicts the image as normal; if closer to 1, it indicates pneumonia. This binary classification setup is well-suited for medical image diagnosis where only two outcomes are possible.

**🛠️ Regularization and Stabilization Techniques**

To ensure that the model performs well on unseen data and does not overfit to the training set, two key techniques are incorporated:

* Batch Normalization is used after each convolutional layer to stabilize and accelerate training. It does this by normalizing the activations, reducing internal covariate shift, and making the model less sensitive to weight initialization.

* Dropout, as mentioned earlier, deactivates a random set of neurons during each training iteration. This encourages the model to develop redundant and diverse internal representations, improving its ability to generalize beyond the training dataset.

**🚀 Key Enhancements & Techniques Used**
* Applied data augmentation techniques such as rescaling, horizontal flipping, rotation, and zoom to improve generalization.
* Balanced the training dataset by sampling equal numbers of pneumonia and normal cases to prevent model bias.
* Introduced regularization techniques like Dropout and Batch Normalization to reduce overfitting and stabilize learning.
* Used a custom callback to monitor precision, recall, and F1-score after each epoch for detailed performance tracking.
* Integrated learning rate scheduling using ReduceLROnPlateau to improve convergence.
* Disabled early stopping to allow full training over 50 epochs for better learning.
* Increased input image resolution to 320×320 for richer feature extraction.
* Thoroughly evaluated the model using confusion matrices, metric reports, and training history plots
