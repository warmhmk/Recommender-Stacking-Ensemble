# Recommender System Using Ensemble Models with Random Forest Regression

This project focuses on developing an advanced recommender system by utilizing a combination of collaborative filtering models and a random forest regressor. The approach taken in this project is relatively unique, as such training on multiple prediction models combined with additional user-based features is not often implemented in typical recommender systems. This method allows for the creation of a more accurate prediction model by capturing a broader range of user preferences and item characteristics.

## Overview

The recommender system is designed to predict user ratings for items based on historical data. The project employs various collaborative filtering models, including Singular Value Decomposition (SVD), Non-negative Matrix Factorization (NMF), K-Nearest Neighbors (KNN), and Co-Clustering, to generate predictions. These predictions, along with additional user-specific features, are then used as inputs for a Random Forest regressor, which makes the final prediction.

## Key Features

- **Ensemble Approach**: By combining predictions from multiple collaborative filtering models, the system reduces the bias associated with individual models, resulting in a more robust and accurate recommender system.
  
- **Random Forest Regression**: The use of a Random Forest regressor as the final prediction model allows for the integration of diverse features, including user voting behavior, which enhances the system's ability to generalize to unseen data.

- **Feature Engineering**: The project involves extensive feature engineering, incorporating user-specific data such as average ratings, total votes, and helpful votes, which significantly improves the model's accuracy.

## Technologies Used

- **Python Libraries**: The project utilizes several Python libraries, including `scikit-learn`, `pandas`, `numpy`, `surprise`, and `cuml` for model implementation and data processing.
- **RAPIDS**: RAPIDS libraries are used to accelerate the training process of the Random Forest model on GPU, significantly reducing the computational time required for hyperparameter tuning and model training.
- **Google Colab**: The code is designed to run on Google Colab with a T4 GPU, leveraging the high computational power and large memory availability.

## Instruction for Running the Code

### Prerequisites

- A T4 GPU is required to run this code efficiently.
- High RAM is recommended due to the large size of the datasets and the complexity of the models.

### Steps to Run the Code

1. **Mount Google Drive**: The datasets and code should be stored in Google Drive. Begin by mounting the drive:

    '''
    from google.colab import drive
    drive.mount('/content/drive')
    '''

2. **Install Required Libraries**: Install the necessary libraries, including `implicit`, `scikit-surprise`, and RAPIDS:

    '''
    !pip install implicit
    !pip install scikit-surprise
    !git clone https://github.com/rapidsai/rapidsai-csp-utils.git
    !python rapidsai-csp-utils/colab/pip-install.py
    '''

3. **Data Exploration**: Load and explore the data:

    '''
    train_df = pd.read_csv('/train.csv')
    test_df = pd.read_csv('/test.csv')
    train_df.head()
    '''

4. **Model Training**: The code involves cross-validation to train multiple models (SVD, NMF, KNN, Co-Clustering) and uses their predictions as features for the Random Forest regressor.

5. **Make Predictions**: Once the models are trained, predictions are made on the test dataset, and the results are saved to a CSV file.

    '''
    res = pd.DataFrame({'ID': range(len(y_pred)), 'rating': y_pred})
    res.to_csv('submission.csv', index=False)
    '''

## Dataset

- **train.csv**: Contains user-item ratings, along with metadata such as the number of votes and helpful votes for each rating.
- **test.csv**: Contains a list of items for which ratings need to be predicted for each user.

## Methodology

1. **Cross-Validation**: The training data is split into multiple folds, and each model is trained on different folds to ensure robust predictions.
2. **Feature Collection**: Additional features such as user voting behavior and average ratings are collected and used as inputs for the Random Forest regressor.
3. **Random Forest Regression**: The final model is a Random Forest regressor that combines the predictions from the collaborative filtering models with the additional features to generate the final ratings.

## Results

The final model achieves a test RMSE of 0.87718, demonstrating the effectiveness of combining multiple models and features in a Random Forest regressor for this task.

## Conclusion

This project highlights the potential of ensemble learning in recommender systems. By integrating various models and user-specific features, the system is able to generate more accurate predictions than traditional single-model approaches. The use of Random Forest regression to combine these predictions further enhances the model's performance, making it a powerful tool for real-world recommendation tasks.

