# Neural_Network_Charity_Analysis

## Project Overview
The purpose of this project is to create a neural network machine learning model to create a binary classifier capable of predicting whether applicants for funding from the AlphabetSoup non-profit will be successful if funded.  The data provided contained over 34,000 historical organizations that have received funding over many years.  There were several columns of information provided on the dataset as features to be fed into the  binary classifier.

## Results

### Data Preprocessing

- The target variable for the model is the IS_SUCCESSFUL column in the dataset.  This column provides a 1 if the organization was considered successful after receiving funding from AlphabetSoup and a 0 if considered unsuccessful.

-  The variables considered as features for the model are the following columns:
    - APPLICATION_TYPE
    - AFFILIATION
    - CLASSIFICATION
    - USE_CASE
    - ORGANIZTION
    - STATUS
    - INCOME_AMT
    - ASK_AMT

- The variables which are neither targets nor features and were removed from the input data are the following columns:
    - EIN
    - NAME
    - *SPECIAL_CONSIDERATIONS
    *This column was included as a feature in the original model for deliverable 2 but was removed from the model for deliverable 3.

### Compiling, Training and Evaluating the Model

- For the original model, I selected a model with two hidden layers.  The number of input features was the length of X_train, and I selected 80 nodes for the first hidden layer and 30 nodes for the second hidden layer.  Because the purpose of this model was binary classification, I selected relu activation functions for the hidden layers and a sigmoid activation function for the output layer.

![shot1](link)

- The model did not achieve the target model performance. The model achieved a 72.6% accuracy score.

![shot2](link)

- I made three attempts to optimize the model to achieve the 75% accuracy target performance level.

    - Attempt #1: I removed the SPECIAL_CONSIDERATIONS column, added additional neurons to the hidden layers and increased the number of epochs.

    ![shot3](link)

    Attempt #1 achieved a 72.5% accuracy score

    ![shot4](link)

    - Attempt #2: I kept the SPECIAL_CONSIDERATIONS column removed, binned the ASK_AMT column to reduce the noise in that column for the model, added additional neurons to the hidden layers, added a third hidden layer, and increased the number of epochs (only did 200 epochs this run, which is less than attempt #1, but still greater than the original model).

     ![shot5](link)

     Attempt #2 scored 72.8% on performance

     ![shot6](link)

    - Attempt #3: I kept the SPECIAL_CONSIDERATIONS column removed, kept the ASK_AMT column binned to reduce the noise in that column for the model, added additional neurons to the hidden layers, added a fourth hidden layer, changed the activation functions in the hidden layers and increased the number of epochs (back up to 300).

    ![shot7](link)

    Attempt #3 performed the worst of all the models at 72.2% accuracy

    ![shot8](link)

## Summary

- summary of the results: While none of the models reached the 75% performance mark, optimization attempt #2 performed best at 72.8%.  This model had the SPECIAL_CONSIDERATIONS column removed, the ASK_AMT column binned, three hidden layers with additional neurons, and ran only 200 epochs.  Notably too, this model used relu activation functions for all hidden layers and the sigmoid activation function for the output layer.  I believe that combination of activation functions is ideal.  Perhaps if attempt #3 had maintained the same activation functions as attempt #2, and increased the number of hidden layers, number of neurons, and the number of epochs it may have outperformed attempt #2.

- Ultimately, none of the neural network models achieved the 75% accuracy target, therefore recommend perhaps using a Support Vector Machine (SVM) instead of a neural network machine learning model as the binary classifier to predict if an applicant for AlphabetSoup funding is going to be successful.  SVMs excel at binary classification, whereas neural network machine learning models are capable of multiple outputs beyond a binary classification.  Additionally, SVMs tend to be less prone to overfitting since the way they function is by maximizing the distance between the two groups rather than encompassing all data within a certain boundary.