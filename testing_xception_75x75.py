import os
import random
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import precision_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, accuracy_score, recall_score, f1_score

# Load the saved model
model_path = "E:\\thesis4000\\xception_2500_75x75.h5"
print("Loading the saved model...")
model = load_model(model_path)
print("Model loaded successfully.")

# Load the test dataset
dataset_dir = r"E:\\thesis4000\\preprocessed_random_500_75px"
print("Loading the test dataset...")

# Load the preprocessed dataset
image_paths_0 = [os.path.join(dataset_dir, "0", img) for img in os.listdir(os.path.join(dataset_dir, "0"))]
image_paths_1 = [os.path.join(dataset_dir, "1", img) for img in os.listdir(os.path.join(dataset_dir, "1"))]

# Combine the image paths and create corresponding labels (0 for "0" folder, 1 for "1" folder)
dataset_images = image_paths_0 + image_paths_1
dataset_labels = np.concatenate((np.zeros(len(image_paths_0)), np.ones(len(image_paths_1))))
print("Dataset loaded successfully.")

# Convert the dataset to numpy arrays
preprocessed_images = np.array(dataset_images)

# Preprocess the images in the dataset for evaluation
preprocessed_images = np.array([img_to_array(load_img(img, target_size=(75, 75))) for img in dataset_images])

# Perform predictions on the preprocessed dataset
print("Performing predictions on the preprocessed dataset...")
predicted_labels = model.predict(preprocessed_images)
predicted_labels = np.argmax(predicted_labels, axis=1)
print("Predictions completed successfully.")

# Calculate evaluation metrics
accuracy = accuracy_score(dataset_labels, predicted_labels)
precision = precision_score(dataset_labels, predicted_labels)
recall = recall_score(dataset_labels, predicted_labels)
f1 = f1_score(dataset_labels, predicted_labels)

# Calculate AUC-ROC
auc_roc = roc_auc_score(dataset_labels, predicted_labels)

# Calculate ROC curve
fpr, tpr, _ = roc_curve(dataset_labels, predicted_labels)

# Calculate precision-recall curve
precision, recall, _ = precision_recall_curve(dataset_labels, predicted_labels)

# Calculate confusion matrix
confusion_mat = confusion_matrix(dataset_labels, predicted_labels)
tn, fp, fn, tp = confusion_mat.ravel()

# Print the evaluation metrics
print("Evaluation Metrics:")
print(f"AUC-ROC: {auc_roc}")
print("ROC Curve:")
print(f"False Positive Rate: {fpr}")
print(f"True Positive Rate: {tpr}")
print("Precision-Recall Curve:")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print("Confusion Matrix:")
print(f"True Negative: {tn}")
print(f"False Positive: {fp}")
print(f"False Negative: {fn}")
print(f"True Positive: {tp}")
