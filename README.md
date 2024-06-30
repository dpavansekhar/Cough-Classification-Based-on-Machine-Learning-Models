# Audio Classification Project

This project focuses on classifying audio signals into two categories: cough and non-cough. The project involves feature extraction from audio files, data preprocessing, training various machine learning models, and evaluating their performance. Additionally, hyperparameter tuning has been performed to improve the models' accuracy.

## Table of Contents

1. [Feature Extraction and Data Preprocessing](#feature-extraction-and-data-preprocessing)
2. [Random Forest Classifier](#random-forest-classifier)
3. [Support Vector Machine](#support-vector-machine)
4. [Gradient Boosting Classifier](#gradient-boosting-classifier)
5. [Convolutional Neural Networks](#convolutional-neural-networks)
6. [Prediction](#prediction)

## Feature Extraction and Data Preprocessing

The process starts with extracting features from audio files. The features extracted include Zero Crossing Rate, Spectral Centroid, Spectral Bandwidth, Spectral Contrast, Spectral Rolloff, RMS Energy, and MFCCs (Mel Frequency Cepstral Coefficients). The extracted features are stored in a CSV file.

The dataset is then balanced by resampling the majority class to match the number of samples in the minority class. This ensures that the model is not biased towards any class.

## Models Used in Audio Classification

### Random Forest Classifier

**Overview:**
The Random Forest classifier is an ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes of the individual trees. It is effective for classification tasks and handles both numerical and categorical data well.

**Implementation:**
1. **Feature Extraction and Data Preprocessing:** 
   - Extract features like Zero Crossing Rate, Spectral Centroid, Spectral Bandwidth, Spectral Contrast, Spectral Rolloff, RMS Energy, and MFCCs from audio files.
   - Preprocess the data by normalizing features and balancing the dataset using resampling techniques if necessary.

2. **Model Training:**
   - Train the Random Forest classifier using the extracted and preprocessed features.
   - Tune hyperparameters such as the number of trees and maximum depth to optimize model performance.

3. **Evaluation:**
   - Evaluate the model using metrics such as accuracy, precision, recall, and F1 score.
   - Cross-validation can be used to ensure the model's generalizability.

### Support Vector Machine (SVM)

**Overview:**
Support Vector Machine is a supervised learning model used for classification tasks. It works by finding the hyperplane that best separates classes in a feature space. SVMs are effective in high-dimensional spaces and when there is a clear margin of separation between classes.

**Implementation:**
1. **Feature Extraction and Data Preprocessing:** 
   - Same as for the Random Forest classifier.

2. **Model Training:**
   - Train the SVM model using the extracted and preprocessed features.
   - Choose appropriate kernel functions (linear, polynomial, radial basis function) based on the data characteristics.

3. **Evaluation:**
   - Evaluate the SVM model using metrics similar to those for the Random Forest classifier.
   - Adjust parameters like C (regularization parameter) and gamma (kernel coefficient) for optimal performance.

### Gradient Boosting Classifier

**Overview:**
Gradient Boosting is an ensemble learning technique that builds multiple decision trees sequentially. It corrects errors made by previous models and combines them to make the final prediction. It is effective in reducing bias and variance in complex datasets.

**Implementation:**
1. **Feature Extraction and Data Preprocessing:** 
   - Same as for the Random Forest classifier.

2. **Model Training:**
   - Train the Gradient Boosting classifier using the extracted and preprocessed features.
   - Adjust hyperparameters such as learning rate, maximum depth of trees, and number of estimators (trees).

3. **Evaluation:**
   - Evaluate the Gradient Boosting classifier using metrics similar to those for the Random Forest classifier.
   - Use techniques like early stopping to prevent overfitting and improve training efficiency.

### Convolutional Neural Networks (CNN)

**Overview:**
CNNs are deep learning models particularly effective for image and audio classification tasks. They use convolutional layers to automatically extract features from input data. Pooling layers reduce dimensionality, and dense layers perform classification based on extracted features.

**Implementation:**
1. **Feature Extraction and Data Preprocessing:** 
   - Extract spectrogram images from audio files.
   - Normalize and preprocess spectrogram images for input to the CNN.

2. **Model Training:**
   - Design a CNN architecture consisting of convolutional layers, pooling layers, and dense layers.
   - Train the CNN model using spectrogram images and corresponding labels (cough or non-cough).

3. **Evaluation:**
   - Evaluate the CNN model using metrics such as accuracy, precision, recall, and F1 score.
   - Fine-tune the model by adjusting hyperparameters like learning rate, batch size, and dropout rates.

### Prediction

After training and evaluating each model, predictions can be made on new audio signals to classify them as cough or non-cough. Ensure that the trained models are saved for deployment and further use in real-world applications.

Each model type offers distinct advantages and may perform differently depending on the characteristics of your audio data. Experimentation with different models and tuning of hyperparameters will help achieve the best performance for your specific classification task.
