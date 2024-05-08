# QTM 347 - Final  Project: Diabetes Health Indicators
This repository contains code and resources for predicting the likelihood of diabetes onset using machine learning models. Diabetes is a chronic condition with significant health implications, and early detection plays a crucial role in the effective management and prevention of complications. By leveraging machine learning techniques on relevant datasets, we aim to develop predictive models to identify individuals at risk of diabetes. We are using data found on Kaggle called the Diabetes Health Indicators Dataset with data collected from the CDC.

Using this dataset, we want to answer the following question:
Which model can most accurately determine the factors (top 3) that predict whether an individual has diabetes or not?

## Using the dataset
The dataset can be downloaded using the link below. When downloading the file, there will be 3 datasets. For the purposes of our project, we used the dataset labeled "diabetes_binary_health_indicators_BRFSS2015" because it splits the diabetes outcome variable into a binary outcome with 0 being "does not have diabetes" and 1 being "for prediabetes or diabetes". The dataset does not contain any N/A values, so it does not require extra cleaning.

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


## Models utilized:
- K-Nearest Neighbors (KNN)
- LASSO Regression
- Best Subset Selection
- Forward Selection
- Principal Component Regression (PCR)
- Decision Tree Classifier
- Neural Network: Multi-Layer Perceptron

## KNN



## LASSO Regression


## Best Subset Selection



## Forward Selection



## Principal Component Regression (PCR)



## Decision Tree Classifer


## Neural Network: Multi-Layer Perceptron

  
