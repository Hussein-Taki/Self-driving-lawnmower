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

def load_and_preprocess_data():
    # Verify if the file exists
    if not os.path.exists(labels_file):  # os.path.exists() checks if a file or directory exists
        print(f"Error: The file '{labels_file}' does not exist.")
        return None, None, None, None
    else:  # If the file exists, load the CSV file
        # Load the CSV file without specifying column names
        try:
            labels_df = pd.read_csv(labels_file, header=None, skiprows=1) # skips the first row as its not part of the data
            print("CSV file loaded successfully.")
            # Display the first few rows of the dataframe to verify its structure
            print(labels_df.head())  # .head displays the first 5 rows of the dataframe
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None, None, None, None

        # Split the combined column into filename and label
        labels_df[['filename', 'label']] = labels_df[0].str.split(';', expand=True)

        # Drop the original combined column
        labels_df = labels_df.drop(columns=[0])

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
                if pd.isna(row['filename']) or pd.isna(row['label']):
                    continue
                img_path = os.path.join(dataset_path, row['filename'])
                print(f"Loading image: {img_path}")
                img = cv2.imread(img_path)
                if img is not None:
                    print(f"Image shape: {img.shape}")  # Debugging: print shape of each loaded image
                    filenames.append(row['filename'])
                    images.append(img)
                    label = 1 if row['label'] >= threshold else 0
                    labels.append(label)
                else:
                    print(f"Failed to load image: {img_path}")

            return images, np.array(labels), filenames  # Return images as a list

        # Load original images with labels and filenames
        original_images, labels, filenames = load_images_with_labels(dataset_path, labels_df, threshold)

        # Print some information about the loaded data
        print(f"Number of images loaded: {len(original_images)}")
        if original_images:
            print(f"Shape of first image: {original_images[0].shape}")
            print(f"Shape of last image: {original_images[-1].shape}")
        print(f"Number of labels: {len(labels)}")
        print(f"Number of filenames: {len(filenames)}")

        # Preprocess images (resize, flatten, normalize)
        def preprocess_images(images):
            processed_images = []  # Create an empty list to store processed images
            for img in images:  # Iterate over each image
                img = cv2.resize(img, (64, 64))  # Resize images to 64x64 pixels
                img = img.flatten() / 255.0  # Flatten the image and normalize pixel values
                processed_images.append(img)  # Append the processed image to the list
            return np.array(processed_images)  # Convert the list to a NumPy array and return it

        # Preprocess the loaded images
        processed_images = preprocess_images(original_images)

        # Split the dataset into training and testing sets
        # random_state=42 is to ensure reproducibility
        X_train, X_test, y_train, y_test = train_test_split(processed_images, labels, test_size=0.2, random_state=42)

        # Print shapes to verify
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")

        return X_train, X_test, y_train, y_test

# You can call the function like this:
# X_train, X_test, y_train, y_test = load_and_preprocess_data()