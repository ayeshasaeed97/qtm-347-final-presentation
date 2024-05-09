# QTM 347 - Final  Project: Diabetes Health Indicators
This repository contains code and resources for predicting the likelihood of diabetes onset using machine learning models. Diabetes is a chronic condition with significant health implications, and early detection plays a crucial role in the effective management and prevention of complications. By leveraging machine learning techniques on relevant datasets, we aim to develop predictive models to identify individuals at risk of diabetes. We are using data found on Kaggle called the Diabetes Health Indicators Dataset with data collected from the CDC.

Dataset link: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data

Using this dataset, we want to answer the following question:
Which model can most accurately determine the factors (top 3) that predict whether an individual has diabetes or not?

## Using the dataset
The dataset can be downloaded using the link [here](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data). When downloading the file, there will be 3 datasets. For the purposes of our project, we used the dataset labeled "diabetes_binary_health_indicators_BRFSS2015" because it splits the diabetes outcome variable into a binary outcome with 0 being "does not have diabetes" and 1 being "for prediabetes or diabetes". The dataset does not contain any N/A values, so it does not require extra cleaning.

### Features:

| Diabetes Binary  | Health Status (self-reported) | 0=No Diabetes 1=Diabetes/Pre-Diabetes     | dependent variable   |
|---------|-------------------------------|---------------------------------------------------|----------------------|
| HighBP    | High Blood Pressure                  | 0 = no highBP, 1 = has highBP                                       | independent variable |
| HighChol    | Have you EVER been told by a doctor, nurse or other health professional that your blood cholesterol is high? | 0 = No, 1 = Yes                                            | independent variable |
| CholCheck  | Cholesterol check within past five years     | 0 = No, 1 = Yes | independent variable |
| BMI     | Body Mass Index                           | Float                              | independent variable |
| Smoker    | Have you smoked at least 100 cigarettes in your entire life?  | 5 packs = 100 cigarettes                    | independent variable |
| Stroke | (Ever told) you had a stroke | 0 = No, 1 = Yes                          | independent variable |
| Heartdiseaseorattack | (Ever told) you had a stroke| 0 = No, 1 = Yes                           | independent variable |
| Physactivity    | doing physical activity or exercise during the past 30 days other than their regular job | 0 = No, 1 = Yes                                       | independent variable |
| Fruits     | Consume Fruit 1 or more times per day                           | 0 = No, 1 = Yes                                         | independent variable |
| Veggies  | Consume Vegetables 1 or more times per day        | 0 = No, 1 = Yes | independent variable |
| HvyAlcoholConsump     | Adult men having more than 14 drinks per week and adult women having more than 7 drinks per week                           | 0 = No, 1 = Yes                              | independent variable |
| AnyHealthcare    | Racial categories             | White, Black, Asian and Others                    | independent variable |
| NoDocBcCost | Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? | 0 = No, 1 = Yes                          | independent variable |
| GenHlth | In general your health is           | 1 is Excellent -> 5 is Poor                          | independent variable |
| MenHlth    | For how many days during the past 30 days was your mental health not good? | In days(0-30);  0 -> (no bad mental health days)                                      | independent variable |
| PhysHlth     | For how many days during the past 30 days was your physical health not good                          | In days(0-30);  0 -> (no bad mental health days)                                           | independent variable |
| DiffWalk  | Do you have serious difficulty walking or climbing stairs? | 0 = No, 1 = Yes  | independent variable |
| Sex    | Sex                           | 1 = Male, 0 = Female                              | independent variable |
| Age    | 13 age categories             | Increments of 5 starts with [18, 24]                    | independent variable |
| Education | Highest grade or year of school completed(1-6) |  1 (never attended school or kindergarten only) --> 6 (college 4 years or more) | independent variable |
| Income | Total income  (1 -8)  | 1 (less than $10,000) ->  8 (being $75,000 or more)        | independent variable |


### Exploratory Data Analysis 

The dataset is cleaned and no outliers were identified that would skew our results. We ran an Ordinary Least squares regression without worrying about confounders just to see the features we're working with and the correlation with the dependent variable, Diabetes Binary. 
![Screenshot 2024-04-28 at 6 53 04 PM](https://github.com/ayeshasaeed97/qtm-347-final-presentation/assets/122938409/d9e0229e-bcea-4bdd-b997-f77b50c5b54e)

From the OLS, HighBP, HighChol, CholCheck and BMI (in that particular order) seem to be the features with the strongest correlation coeeficients. 

![Screenshot 2024-05-08 at 3 53 27 PM](https://github.com/ayeshasaeed97/qtm-347-final-presentation/assets/122938409/430f374c-25f4-4360-bf89-4b47a880e6d2)

As for the heat map, GenHlth and HighBP seem to have the strongest correlation with Diabetes Binary. These give us a good point of reference for the other regression models. 

## Models utilized:
- K-Nearest Neighbors (KNN)
- LASSO Regression
- Best Subset Selection
- Forward Selection
- Principal Component Regression (PCR)
- Decision Tree Classifier
- Neural Network: Multi-Layer Perceptron

(add a comment about how the data was split into training and test set) 

## KNN
To run our k-Nearest Neigbors classification, we used KNeighborRegressor from sklearn.neighbors after splitting our dataset into training and test data. We used cross-validation to find the best n_neighbors value which turned out to be n=12 with an accuracy of 0.85842 and a test MSE of 0.1416. However, we also calculated the accuracy score and test MSE for n values ranging from 1-13 to see if other values of n were within 2 standard deviation of the n=12 score. This was, we could get similar predictive power with a lower n. We saw the n=8 and n=10 have very close test MSEs of 0.1430 and 0.1423, respectively. The accuracy score for n=8 was 0.85702 and for n=10 it was 0.85768.

<img width="417" alt="Screenshot 2024-05-08 at 9 28 29 PM" src="https://github.com/ayeshasaeed97/qtm-347-final-presentation/assets/98335777/faa797fa-fc5f-41e8-90ee-66d3771ea94e">

<img width="519" alt="Screenshot 2024-05-08 at 8 34 58 PM" src="https://github.com/ayeshasaeed97/qtm-347-final-presentation/assets/98335777/5e708aea-cd0f-4ff0-8d7b-ba7ccae4b95a">

## LASSO Regression

The pre-processing for the lasso included selcting the predictor variables from the training and test dataset (X_train and X_test), and exluding the first column which is the dependent variable. We then defined a 5-fold cross-validation (kfold) method to select the optimal lasso regularization parameter (alpha). We initialized an ElasticNetCV model (lassoCV) with 100 alphas, L1 ratio of 1 (for Lasso), and the cross-validation method. Next, a pipeline (pipeCV) was created consisting of a StandardScaler and the ElasticNetCV model, and fit it to the training data (X_train_reg, y_train) with only the predictors.  After retrieving the tuned alpha value from the fitted ElasticNetCV model, we initialized a Lasso regression model using the tuned alpha value. All this was done without standardizing the input features, but Lasso puts constraints on the size of the coefficients of the features and that is dependent upon the magnitude of each variable, which makes it necessary. To standardize the variables, we initialized a StandardScaler (scaler) and created a pipeline (pipe) that contains both the scaler and Lasso. Finally, we fit the new pipeline to the training data. 


**The top 3 features identified are: GenHlth, BMI, HighBP, with accuracy Score of 0.86** 

![Screenshot 2024-05-08 at 7 21 09 PM](https://github.com/ayeshasaeed97/qtm-347-final-presentation/assets/122938409/00f2c6a3-a82a-4c24-b55b-13c700afcded)

The coefficients represent the effect of each predictor variable on the target variable in the Lasso regression model. Positive coefficients indicate a positive relationship with the target variable, while negative coefficients indicate a negative relationship. GenHlth, BMI, HighBP, and HighChol have the largest positive coefficients, suggesting that individuals with higher general health ratings, higher BMI, and those with high blood pressure or high cholesterol are more likely to have the target outcome. Regarding the fact that the Lasso regression did not shrink any coefficients to zero, it suggests that all predictors included in the model contribute at least some predictive power for the target variable. This can be interpreted as indicating that each of these predictors contains valuable information for predicting the outcome. However, it's important to note that this lack of shrinkage to zero may also imply that the model is complex and potentially overfitting the data. 


## Best Subset Selection



## Forward Selection



## Principal Component Regression (PCR)



## Decision Tree Classifer
For the decision tree classifier, we changed the Diabetes_binary indicator (indicated 1 for diabetes/pre-diabetes and 0 for no diabetes) as "Yes" and "No", respectively. Afterwards, using DecisionTreeClassifier from sklearn.tree, we fitted our model to our raw data. Using that, we calculated the accuracy score of 0.8624. Next, we did a train-test split and calculated accuracy scores for our decision tree across max_depths ranging from 1 to 10. Both n=5 and n=6 give scores of 0.8636. For the tree, we used a max_depth of n=5 to save on computing time.

<img width="566" alt="Screenshot 2024-05-08 at 6 55 07 PM" src="https://github.com/ayeshasaeed97/qtm-347-final-presentation/assets/98335777/191710cc-7801-445a-8ce1-c7ea31645d9f">

Below is the pre-pruned decision tree.
![output dt-pre-pruned](https://github.com/ayeshasaeed97/qtm-347-final-presentation/assets/98335777/9c980c9e-e39d-4b35-aa42-ef8ddd5b3ea2)

While it not easily seen in the decision tree, below you can see that most features give No-No or Yes-Yes for classes. This indicates high node purity which we will attempt to fix by pruning the tree. 

<img width="541" alt="Screenshot 2024-05-08 at 6 57 56 PM" src="https://github.com/ayeshasaeed97/qtm-347-final-presentation/assets/98335777/50f0777f-2d51-42b3-9fe9-ca2592908ce6">

Pruning the tree got rid our problem with node purity because we did get features that showed classification into Yes or No for whether and individual was predicted to have diabetes given the features. We got an improved accuracy score of 0.866, but this was not a large improvement compared to our previous accuracy score of 0.8636. 
A note: pruning the decision tree was computationally expensive to run, with it taking about 26 minutes to run. 

![dt post-pruned](https://github.com/ayeshasaeed97/qtm-347-final-presentation/assets/98335777/0340a8e8-a8de-4363-adcc-b7f0c81a8db0)

<img width="512" alt="Screenshot 2024-05-08 at 7 00 12 PM" src="https://github.com/ayeshasaeed97/qtm-347-final-presentation/assets/98335777/5d527978-d94e-4038-b790-13807df35593">

From the decision tree, we see the following are the top 3 most important features in determining whether an individual has diabetes or not: HighBP, GenHlth, and BMI. 

<img width="271" alt="Screenshot 2024-05-08 at 7 04 47 PM" src="https://github.com/ayeshasaeed97/qtm-347-final-presentation/assets/98335777/8d7b0406-b85a-4462-846e-823c1dac9477">

Producing a confusion table allows us to see the misclassification or Type I and Type II errors. Looking at the table, the model makes a lot of false negative predictions. This could be caused by the dataset not having an even amount of observation for each outcome (1 or 0).
<img width="195" alt="Screenshot 2024-05-08 at 8 08 46 PM" src="https://github.com/ayeshasaeed97/qtm-347-final-presentation/assets/98335777/0288b858-f1fa-44ab-aab8-9e20d43e9b20">


## Neural Network: Multi-Layer Perceptron

Since diabetes prediction is a binary classification task, a deep learning model like multi-layer perceptron (MLP) is used. MLPs are versatile and can handle a wide range of input data types, including tabular data like the Pima dataset. They can learn complex non-linear relationships between features and the target variable. Likewise, MLPs can scale well to large datasets and high-dimensional feature spaces, making them suitable for this large dataset. Additionally, MLP mplementations are also readily available in popular machine learning libraries like scikit-learn, TensorFlow, and Keras, making them easy to use and experiment with.

First, the diabetes dataset is preprocessed by feature extraction. In one dataframe, all the feature values are stored. In another dataframe, the ‘Outcome’ or the predicted variable is stored. The dataset is then split into a training and test sets. The features are normalized. The test set is further split to include a validation set. Then, the neural network is defined using the Keras library with a TensorFlow backend. This neural network architecture consists of three dense layers with batch normalization, ReLU activation functions, and dropout regularization between each pair of layers. The output layer uses a sigmoid activation function for binary classification.  The sigmoid function, also known as the logistic function, maps the output of the neural network to a value between 0 and 1, representing the probability of the positive class (class 1).

Following that, the neural network model is compiled using the binary cross-entropy loss function, the Adam optimizer, and accuracy as the evaluation metric. Adam is a popular choice for optimization as it combines techniques like momentum and adaptive learning rates to efficiently capture model parameters during training. The neural network model is trained using a specified number of epochs (200). It also utilizes validation data to monitor the model's performance during training. Learning rate reduction and early stopping strategies are also applied to  improve training efficiency and prevent overfitting. The model's performance is evaluated on the validation data after each epoch to monitor its progress.

Another cross-validation (5-fold) is performed to asses the model performance and generalization ability using an ‘MLPClassifier’ from ‘sklearn.neural_network’, ‘StandardScaler’from ‘sklearn.preprocessing’, and ‘make_pipeline’ from ‘sklearn.pipeline’. For further performance evaluation, an ROC (Receiver Operating Characteristic) curve is used. ROC curves provide a comprehensive view of how well the model can distinguish between two classes. It captures how the model's sensitivity (true positive rate) and specificity (true negative rate) vary with different threshold values. 

### The model summary shows that: 

### Training Set:
- Accuracy: 86.8%
- Precision:
    Class 0 (No diabetes): 88%
    Class 1 (Diabetes): 61%
- Recall:
    Class 0: 99%
    Class 1: 14%
- F1-score:
    Class 0: 93%
    Class 1: 22%
- Support:
    Class 0: 174,732
    Class 1: 28,212

### Validation Set:
- Accuracy: 86.4%
- Precision:
    Class 0 (No diabetes): 87%
    Class 1 (Diabetes): 60%
- Recall:
    Class 0: 99%
    Class 1: 13%
- F1-score:
    Class 0: 93%
    Class 1: 21%
- Support:
    Class 0: 34,834
    Class 1: 5,754

###  Test Set:
- Accuracy: 86.9%
- Precision:
    Class 0 (No diabetes): 88%
    Class 1 (Diabetes): 58%
- Recall:
    Class 0: 98%
    Class 1: 14%
- F1-score:
    Class 0: 93%
    Class 1: 23%
- Support:
    Class 0: 8,768
    Class 1: 1,380

### AUC Score: 0.83

![image](https://github.com/ayeshasaeed97/qtm-347-final-presentation/assets/90111688/c11ecd62-7591-4990-a60e-75b558e56c1b)


The neural network demonstrates consistent performance across the training, development, and test sets, with accuracies ranging from 86.4% to 86.9%. This indicates that the model generalizes well to unseen data. However, the dataset seems to suffer from class imbalance, as indicated by the discrepancy between precision, recall, and F1-score values for the minority class (Class 1/Diabetes). While the model performs well in identifying the majority class (Class 0/No Diabetes), it struggles with the minority class, particularly in terms of recall.

The precision for the minority class is higher than its recall, suggesting that when the model predicts an instance as positive (Class 1/Diabetes), it is often correct, but it misses many positive instances. This trade-off between precision and recall needs to be considered based on the specific application requirements.

The F1 score balances precision and recall, providing a single metric to evaluate the model's performance. However, it remains lower for the minority class compared to the majority class, indicating the need for further improvement, possibly through techniques like class weighting, oversampling, or adjusting the decision threshold.

An AUC of 0.83 suggests that your model performs significantly better than random guessing. It correctly ranks 83% of randomly chosen positive instances higher than randomly chosen negative instances.

Ultimately, the model performs well in terms of accuracy, it seems to struggle with correctly identifying instances of diabetes (class 1), as indicated by the lower precision, recall, and F1-score for this class compared to class 0. This suggests that the model might benefit from further optimization or additional data preprocessing techniques to improve its performance, especially for the minority class. In this context, the minority class refers to the class with fewer instances or samples compared to the majority class.

Reference Code: https://www.kaggle.com/code/kredy10/simple-neural-network-for-diabetes-prediction
https://www.kaggle.com/code/kanncaa1/roc-curve-with-k-fold-cv

## Conclusion

### Table of results

| Model | Accuracy Score   |
| ----- | ---------------- |
| KNN |         0.85842    |
| Lasso |      0.85974     |
| PCR |      0.86246       |
| Decision Tree | 0.86645  |
| Neural Network | 0.869   |

So we can see that decision tree and neural network had the highest accuracy scores of 0.86645 and 0.869, respectively. Intuitively, decision tree is the best model for answering our question because it closely mimics human decision making. It is similar to how a healthcare provider would run tests and look at results to determine whether a patient has diabetes. It is also a model we could scale-up into an actually effective tool for individuals and healthcare providers to keep track of their own health. 

However, when we look at the computational cost of running these models, especially on a large scale, they may not be the best options. Pruning the decision tree alone took almost 30 minutes. The similar accuracy score indicates PCR performed similarly to decision tree and neural network. It is a simpler model to run, but possibly less intuitive for people without a quantitative background. The key with the decision tree would be to take the model and create an application that allows individuals to put in their test scores for the various features we highlighted in our analysis (blood pressure, BMI, general health rating).



