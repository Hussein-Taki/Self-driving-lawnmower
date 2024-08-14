# Autonomous Lawn Mower Project

## Overview

The AI Autonomous Lawn Mower Project aims to create a fully functional, autonomous lawnmower. The project is a comprehensive collection of code and resources, ranging from machine learning classifiers to control systems implemented on Raspberry Pi/Arduino. The project is modular, allowing each component to be developed and tested independently, before integration into the complete system.

## Disclaimer
This project is still in progress
Currently includes: Grass classifier models

## Project Structure

The project is (as of now) organized into the following directories and files:

GrassClassifier/
│
├── algorithms/
│ ├── lg.py # Logistic Regression implementation
│ ├── k_nearest.py # K-Nearest Neighbors implementation
│ └── svm.py # Support Vector Machine implementation
│
├── preprocessing/
│ └── preprocess.py # Data loading and preprocessing functions
│
├── data/
│ ├── training/ # Directory containing training images
│ │ └── image/ # Image files for training
│ └── test/ # (Optional) Directory containing test images
│
├── main.py # Main script to run the chosen algorithm
├── requirements.txt # List of dependencies required to run the project
└── README.md # Project overview and documentation


## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.7 or higher
- Pip (Python package manager)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Hussein-Taki/Self-driving-lawnmower.git
   cd Self-driving-lawnmower

2. **Install required dependencies**

     ```bash
     pip install -r requirements.txt

3. **Prepare your dataset:**

Place your training images in the data/training/image directory.
Ensure your labels are in a CSV file (Grass.csv) located in data/training/.


## Usage

### Running the Classification Algorithms

You can run different classification algorithms by specifying the `--method` parameter:

#### Logistic Regression:

    ```bash
    python main.py --method lg

#### K-Nearest Neighbors (KNN):
    python main.py --method knn

#### Support Vector Machine (SVM):
     python main.py --method svm

#### Convolutional Neural Network (CNN):
    python main.py --method cnn


## Customizing Parameters
You can modify parameters like the number of neighbors for KNN or the C parameter for SVM by editing the respective files in the algorithms/ directory.

## Project Details
Data Preprocessing
Data is preprocessed using functions defined in preprocessing/preprocess.py. The images are loaded, resized, and normalized before being split into training and testing datasets.
Model Evaluation
After training, each model's performance is evaluated using metrics such as Accuracy, Precision, Recall, and F1 Score.
A confusion matrix is also displayed to visualize the classification performance.
Contributing
If you would like to contribute to this project, feel free to fork the repository and submit a pull request.

## Contact
For any inquiries or suggestions, please contact Hussein Taki at h.taki@student.reading.ac.uk
