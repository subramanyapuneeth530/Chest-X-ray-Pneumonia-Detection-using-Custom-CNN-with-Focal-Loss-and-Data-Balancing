# 🗂️ Dataset Overview
This project uses the Chest X-Ray Images (Pneumonia) dataset, publicly available from Kaggle. It contains X-ray images classified into two categories: NORMAL (healthy lungs) and PNEUMONIA (lungs showing signs of infection).

The dataset is organized into three original folders:

``` train ``` Contains the majority of images used for training.

``` val ``` A small validation set used for tuning the model.

```test``` Used to evaluate model performance on unseen data.

Each of these folders has two subdirectories:

```NORMAL``` Images of healthy lungs.

```PNEUMONIA``` Images showing pneumonia infection.

## ⚖️ Why a Balanced Dataset?
The original dataset had a significant class imbalance, with the number of pneumonia images being much higher than normal cases. Training a model on this unbalanced data could lead to biased predictions, where the model learns to favor the majority class (PNEUMONIA) and fails to generalize well on the minority class (NORMAL).

To mitigate this issue, we created a new folder called:

```balanced_train``` Contains an equal number of NORMAL and PNEUMONIA images randomly sampled from the original train/ folder.

Balancing the training dataset helps:
* Improve the model’s ability to recognize both classes equally.
* Prevent overfitting to the dominant class.
* Boost overall evaluation metrics like F1-score, Precision, and Recall, especially for the minority class.

# link for the dataset 
```https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia```
