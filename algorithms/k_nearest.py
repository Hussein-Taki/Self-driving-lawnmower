import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import time
import sys
import os
import matplotlib.pyplot as plt
from preprocess import load_and_preprocess_data



def train_evaluate_knn(X_train, X_test, y_train, y_test):
    # Initialize the knn model
    # KNeighborsClassifier creates an instance of the model
    # "n_neighbours" paramter is the number of neighbours to consider
    knn_model = KNeighborsClassifier(n_neighbors=3)

    # train the model
    # X_train is the input data
    # y_train is the target data
    start_time = time.time() # record start time before training
    knn_model.fit(X_train, y_train)
    training_time = time.time() - start_time # calculate training time

    #predict the output for the test data (x_test)
    start_time = time.time() # record start time before prediction
    knn_y_pred = knn_model.predict(X_test)
    prediction_time = time.time() - start_time # calculate prediction time

    # Evaluate the model
    knn_accuracy = accuracy_score(y_test, knn_y_pred)  # Calculate the accuracy
    knn_precision = precision_score(y_test, knn_y_pred) # Calculate the precision
    knn_recall = recall_score(y_test, knn_y_pred) # Calculate the recall
    knn_f1 = f1_score(y_test, knn_y_pred) # Calculate the F1 score

    # print evaluations metrics to 4 decimal places
    print(f"KKN Accuracy {knn_accuracy:.4f}") # print accuracy
    print(f"KNN Precision: {knn_precision:.4f}") # print precision
    print(f"KNN Recall: {knn_recall:.4f}") # print recall
    print(f"KNN F1 Score: {knn_f1:.4f}") # print F1 score
    print(f"KNN Training Time: {training_time:.4f} seconds") # print training time
    print(f"KNN Prediction Time: {prediction_time:.4f} seconds") # print prediction time

    # Display confusion matrix
    knn_cm = confusion_matrix(y_test, knn_y_pred) # generate confusion matrix
    # Create a ConfusionMatrixDisplay object to visualize the cm
    knn_disp = ConfusionMatrixDisplay(confusion_matrix=knn_cm, display_labels=['Not Grass', 'Grass'])
    #plot the confusion matrix
    knn_disp.plot()
    #show the plot
    plt.show()

