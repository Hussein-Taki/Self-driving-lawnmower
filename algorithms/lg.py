import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import time
import matplotlib.pyplot as plt
import sys
from preprocess import load_and_preprocess_data


def run_logistic_regression(X_train, X_test, y_train, y_test):
    # Train Logistic Regression model
    model = LogisticRegression(max_iter=1000)  # Initialize a Logistic Regression model with a maximum of 1000 iterations
    start_time = time.time()
    model.fit(X_train, y_train)  # Train the model using the training data
    training_time = time.time() - start_time

    # Make predictions on the test set
    start_time = time.time()
    lr_y_pred = model.predict(X_test)  # Use the trained model to make predictions on the test data
    prediction_time = time.time() - start_time

    # Evaluate the model
    accuracy = accuracy_score(y_test, lr_y_pred)  # Calculate the accuracy of the model
    precision = precision_score(y_test, lr_y_pred)  # Calculate the precision of the model
    recall = recall_score(y_test, lr_y_pred)  # Calculate the recall of the model
    f1 = f1_score(y_test, lr_y_pred)  # Calculate the F1 score of the model

    # Print the evaluation metrics and the time taken for training and prediction
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Training Time: {training_time:.4f} seconds")
    print(f"Prediction Time: {prediction_time:.4f} seconds")

    # Display confusion matrix
    cm = confusion_matrix(y_test, lr_y_pred)  # Generate the confusion matrix for the predictions
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Grass', 'Grass'])  # Create a ConfusionMatrixDisplay object to visualize the confusion matrix
    disp.plot()  # Plot the confusion matrix
    plt.show()  # Show the plot
