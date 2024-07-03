
import sys
import os
import argparse
import preprocess

# Add the algorithms folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithms'))

# Import method scripts
from algorithms.lg import run_logistic_regression
from algorithms.k_nearest import train_evaluate_knn

def main(method):
    # Load and preprocess data
    dataset_path = 'C:\\Users\\husse\\Downloads\\archive\\training\\training\\image'
    labels_file = 'C:\\Users\\husse\\Downloads\\archive\\training\\training\\Grass.csv'
    X_train, X_test, y_train, y_test = preprocess.load_and_preprocess_data(dataset_path, labels_file)
    if method == 'knn':
        train_evaluate_knn( X_train, X_test, y_train, y_test)
    elif method == 'lg':
        run_logistic_regression(X_train, X_test, y_train, y_test)


    elif method == 'svm':
        # Placeholder for SVM (implement similarly to KNN)
        print("SVM method is not implemented yet.")
    else:
        print("Invalid method selected. Please choose from 'knn', 'lg', or 'svm'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run different classification algorithms for self-driving lawn mower.')
    parser.add_argument('--method', type=str, required=True, help="Choose the method to run: 'knn', 'lg', 'svm'")
    args = parser.parse_args()
    main(args.method)




