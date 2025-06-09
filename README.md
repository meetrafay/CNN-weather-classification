# 🌦️ CNN for Weather Classification

This project builds and evaluates convolutional neural network (CNN) models to classify images into 11 different weather conditions using custom and pretrained models.

---

## 📁 Section 1: Dataset Loading and Visualization

- Loaded a weather dataset containing 11 distinct classes.
- Verified data distribution and displayed 2–3 sample images per class using `matplotlib` for visual inspection.

---

## ⚙️ Section 2: Data Preprocessing

- Resized all images to a consistent shape (150x150 or 224x224).
- Normalized pixel values to the range [0, 1] for faster convergence.
- Split the dataset into training, validation, and test sets.
- Applied data augmentation (rotation, zoom, brightness, horizontal flipping) using `ImageDataGenerator`.

---

## 🧠 Section 3: Model Building

### ✅ Base CNN Architecture:
- `Conv2D(32, (3,3), relu)` → `MaxPooling2D`
- `Conv2D(64, (3,3), relu)` → `MaxPooling2D`
- `Flatten` → `Dense(128, relu)` → `Dense(11, softmax)`

### ⚙️ Compilation:
- Loss: `categorical_crossentropy`
- Optimizer: `Adam`
- Metrics: `accuracy`

---

## 🏋️ Section 4: Training and Evaluation

- Trained model with appropriate batch size and epochs.
- Plotted training and validation accuracy/loss across epochs.
- Evaluated performance on test set:
  - **Accuracy**
  - **Classification Report** (Precision, Recall, F1-score)
  - **Confusion Matrix**
- Displayed misclassified images with predicted vs actual labels.

---

## 🔁 Section 5: Model Improvement

### ✅ Enhancements:
- Added **Dropout** and **BatchNormalization** layers to reduce overfitting.
- Re-trained model and compared performance with the base model.

### ✅ Transfer Learning:
- Replaced base CNN with **MobileNetV2** (pretrained on ImageNet).
- Frozen base layers, added custom classification head.
- Fine-tuned model on weather dataset for improved accuracy.

---

## 📊 Final Results

- Compared base CNN and transfer learning models.
- Achieved higher accuracy and better generalization using pretrained MobileNetV2.
