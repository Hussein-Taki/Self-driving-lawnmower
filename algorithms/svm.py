import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import time
import matplotlib.pyplot as plt

# function to train and evaluate the SVM model
def train_evaluate_svm(X_train, X_test, y_train, y_test):
    # Initialize the SVM model
    # SVC creates an instance of the Support Vector Classifier
    # "C" is the regularization parameter, "kernel" specifies the kernel type
    svm_model = SVC(C=1.0, kernel='linear')

    # Train the model
    start_time = time.time() # time.time() returns the current time in seconds
    svm_model.fit(X_train, y_train) # fit() trains the model
    training_time = time.time() - start_time # calculate the training time

    # Predict on X_test
    start_time = time.time()  # Record start time before prediction
    svm_y_pred = svm_model.predict(X_test) # predict() makes predictions on the test data
    prediction_time = time.time() - start_time  # Calculate prediction time

    # Calculate evaluation metrics
    svm_accuracy = accuracy_score(y_test, svm_y_pred)
    svm_precision = precision_score(y_test, svm_y_pred)
    svm_recall = recall_score(y_test, svm_y_pred)
    svm_f1 = f1_score(y_test, svm_y_pred)

    # Print evaluation metrics
    print(f"SVM Accuracy: {svm_accuracy:.4f}")
    print(f"SVM Precision: {svm_precision:.4f}")
    print(f"SVM Recall: {svm_recall:.4f}")
    print(f"SVM F1 Score: {svm_f1:.4f}")
    print(f"SVM Training Time: {training_time:.4f} seconds")  # Print training time
    print(f"SVM Prediction Time: {prediction_time:.4f} seconds")  # Print prediction time

    # Display confusion matrix
    svm_cm = confusion_matrix(y_test, svm_y_pred)  # Generate confusion matrix
    # Create a ConfusionMatrixDisplay object to visualize the cm
    svm_disp = ConfusionMatrixDisplay(confusion_matrix=svm_cm, display_labels=['Not Grass', 'Grass'])
    svm_disp.plot()  # Plot the confusion matrix
    plt.show()  # Show the plot


