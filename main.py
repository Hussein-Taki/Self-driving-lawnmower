import numpy as np
import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import time
import matplotlib.pyplot as plt

# Path to the extracted dataset and CSV file
dataset_path = 'C:\\Users\\husse\\Downloads\\archive\\training\\training\\image'
labels_file = 'C:\\Users\\husse\\Downloads\\archive\\training\\training\\Grass.csv'

# Verify if the file exists
if not os.path.exists(labels_file):
    print(f"Error: The file '{labels_file}' does not exist.")
else:
    # Load the CSV file without specifying column names
    try:
        labels_df = pd.read_csv(labels_file, header=None, skiprows=1)
        print("CSV file loaded successfully.")
        # Display the first few rows of the dataframe to verify its structure
        print(labels_df.head())
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        labels_df = None

        # Load the CSV file without specifying column names
        try:
            labels_df = pd.read_csv(labels_file, header=None, names=['combined'])
            print("CSV file loaded successfully.")
            print(labels_df.head())
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            labels_df = None

        if labels_df is not None:
            # Split the combined column into filename and label
            labels_df[['filename', 'label']] = labels_df['combined'].str.split(';', expand=True)

            # Drop the original combined column
            labels_df = labels_df.drop(columns=['combined'])

            # Convert label to integer
            labels_df['label'] = labels_df['label'].astype(int)

            # Print the first few rows again to verify the column names and split
            print(labels_df.head())

        # Define a threshold to convert scores to binary labels
        threshold = 50

        def load_images_with_labels(dataset_path, labels_df, threshold):
            images = []
            labels = []
            filenames = []
            for index, row in labels_df.iterrows():
                img_path = os.path.join(dataset_path, row['filename'])
                img = cv2.imread(img_path)
                if img is not None:
                    filenames.append(row['filename'])  # Store filename for later verification
                    images.append(img)
                    # Convert the score to a binary label based on the threshold
                    label = 1 if row['label'] >= threshold else 0
                    labels.append(label)
            return np.array(images), np.array(labels), filenames

        # Load original images with labels
        original_images, labels, filenames = load_images_with_labels(dataset_path, labels_df, threshold)

        # Preprocess images (resize, flatten, normalize)
        def preprocess_images(images):
            processed_images = []
            for img in images:
                img = cv2.resize(img, (64, 64))  # Resize images to 64x64 pixels
                img = img.flatten() / 255.0      # Flatten the image and normalize pixel values
                processed_images.append(img)
            return np.array(processed_images)

        processed_images = preprocess_images(original_images)

        # Function to display images
        def display_images(images, titles, cmap=None):
            plt.figure(figsize=(12, 6))
            for i in range(len(images)):
                plt.subplot(1, len(images), i + 1)
                plt.imshow(images[i], cmap=cmap)
                plt.title(titles[i])
                plt.axis('off')
            plt.show()

        # Display some original and preprocessed images for verification
        num_images_to_display = 5
        display_images(original_images[:num_images_to_display], filenames[:num_images_to_display])
        display_images([img.reshape(64, 64, 3) for img in processed_images[:num_images_to_display]], filenames[:num_images_to_display])

        # Prepare the dataset
        X, y = processed_images, labels

