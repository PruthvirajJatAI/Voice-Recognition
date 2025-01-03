# Voice-Recognition
This project focuses on developing a voice recognition system using machine learning techniques. It includes both the dataset and the code used to train and evaluate the model. The dataset consists of voice recordings, while the code processes the data and applies machine learning algorithms to classify or recognize the voices.

# Features
1. Data Integration: Utilizes a CSV dataset containing voice features to train the voice recognition model.

2. Machine Learning Model: Employs various machine learning algorithms to classify and recognize voice patterns.

3. Preprocessing: The data is preprocessed to remove noise and irrelevant features, making it suitable for model training.

4. Model Training and Evaluation: Trains a model on the dataset and evaluates it using standard performance metrics.

5. Real-Time Recognition: The system is designed to recognize voices in real-time (if applicable).

# Work Done
1. Data Collection:
Dataset: Utilized a voice dataset (voice.csv) containing various features extracted from voice recordings.

2. Preprocessing: The CSV data was cleaned, normalized, and preprocessed for feature extraction.

# Model Development:
1. Machine Learning Model: Developed Python scripts to train a voice recognition model using various algorithms such as Support Vector Machines (SVM), Random Forests, or Neural Networks.

2. Feature Engineering: Applied techniques such as feature selection and scaling to improve the model's accuracy.

3. Model Evaluation: Configured performance metrics like accuracy, precision, recall, and F1-score to evaluate the model.


# Prerequisites
1. Python 3.x: Ensure you have Python installed.

2. Python Libraries: Install the following Python libraries:
            numpy, pandas, scikit-learn, matplotlib, seaborn

Dataset: The dataset voice.csv contains the voice features and is used to train the model.

# Code Overview
Parameters 
-  train_size: Proportion of the dataset used for training.

-  random_state: Random seed for reproducibility.

-  n_estimators: Number of trees in the Random Forest model.

# Durations:
- training_time: Time taken for training the model.

- evaluation_time: Time taken to evaluate the model.

# Modeling Logic:
- algorithm: Algorithm used for classification (e.g., RandomForest, SVM).

# Key Functions
- preprocess_data(data): Preprocesses the dataset by scaling and cleaning the data.

- train_model(X_train, y_train): Trains the model on the training set.

- evaluate_model(model, X_test, y_test): Evaluates the trained model using test data.

predict_voice(input_features): Recognizes the voice based on input features.
