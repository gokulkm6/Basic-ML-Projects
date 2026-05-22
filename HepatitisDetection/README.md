**Hepatitis Detection Using SVM and Weka Tool Based PSO**

Overview
1. This project aims to develop a system for the detection of hepatitis using Support Vector Machine (SVM) and Particle Swarm Optimization (PSO). The Weka tool is used for data preprocessing and feature selection, while PSO is employed to optimize the parameters of the SVM model.

Table of Contents
-----------------
1. Introduction
2. Dataset
3. Prerequisites
4. Installation
5. Usage
6. Project Structure
7. Results
8. Contributing
9. License
10. Acknowledgments

Introduction
1. Hepatitis is a significant health issue worldwide. Early and accurate detection can improve patient outcomes and reduce healthcare costs. This project leverages machine learning techniques to develop an efficient and accurate hepatitis detection system.

Objectives
1. To preprocess hepatitis dataset using Weka tool.
2. To select relevant features using Weka's feature selection methods.
3. To optimize SVM parameters using Particle Swarm Optimization (PSO).
4. To evaluate the performance of the SVM model for hepatitis detection.

Dataset
1. The dataset used in this project is the Hepatitis dataset from the UCI Machine Learning Repository. It contains various medical features related to hepatitis patients.

Features
1. Age
2. Sex
3. Malaise
4. Spiders
5. SGOT
6. Albumin

Target Variable
1. Class (1: Die, 2: Live)

Prerequisites
1. Python (for implementing PSO and SVM)
2. Weka 3.8 or later
3. Required Python libraries: numpy, pandas, scikit-learn

Installation
1. Download and install Weka:
https://sourceforge.net/projects/weka/

Clone the repository:
1. git clone https://github.com/GoksGokul/Hepatitis-Detection.git
cd Hepatitis-Detection

Install required Python libraries:
1. pip install numpy pandas scikit-learn

Usage
Data Preprocessing
1. Open Weka GUI.
2. Load the Hepatitis dataset.
3. Perform necessary preprocessing (e.g., handling missing values, normalization).
4. Save the preprocessed dataset.

Feature Selection
1. Use Weka's feature selection methods (e.g., CfsSubsetEval, InfoGainAttributeEval) to select relevant features.
2. Export the selected features for model training.
   
Model Evaluation
1. Evaluate the performance of the optimized SVM model on the test dataset.
python hdspso.py

Results
1. The results of the project include the optimized parameters for the SVM model and the performance metrics (accuracy, precision, recall, F1-score) on the test dataset. These results are stored in the results directory.

Contributing
1. Contributions are welcome! Please fork the repository and submit a pull request.

License
1. This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
1. The UCI Machine Learning Repository for providing the Hepatitis dataset.
2. The developers of Weka and scikit-learn for their excellent tools.
3. All contributors and supporters of this project.
