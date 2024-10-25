# Loan Status Prediction Model  

This project implements a **Loan Status Prediction Model** using **Support Vector Machine (SVM)**, a popular binary classification algorithm. The goal is to predict whether a loan application will be approved or not, based on various applicant details. 

---

## Overview  
The dataset contains applicant details such as income, education, marital status, and property area. The target variable is the **loan status**, indicating whether a loan was approved (1) or not (0). This model aims to provide an automated solution for predicting loan approvals based on applicant profiles.

---

## Dataset Description  
The dataset used for this project was obtained from Kaggle:  
[Loan Prediction Dataset](https://www.kaggle.com/datasets/ninzaami/loan-predication)  

It contains the following features:  

| Feature          | Description                            |
|------------------|----------------------------------------|
| Gender           | Applicant's gender (Male/Female)       |
| Married          | Marital status (Yes/No)                |
| Dependents       | Number of dependents                   |
| Education        | Educational qualification (Graduate/Not Graduate) |
| Self_Employed    | Employment type (Self-employed/Not)    |
| ApplicantIncome  | Income of the applicant                |
| CoapplicantIncome| Income of the co-applicant             |
| LoanAmount       | Loan amount requested                  |
| Loan_Amount_Term | Loan repayment term (in months)        |
| Credit_History   | History of credit repayment (1 = Yes, 0 = No) |
| Property_Area    | Area where the property is located (Urban/Semiurban/Rural) |

---

## Model Performance  
- **Training Accuracy:** 77.86%  
- **Testing Accuracy:** 81.25%  

---

## Usage  
Here’s how you can use the model to predict loan status:

```python
import numpy as np
import pandas as pd

# Input data: ('Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
#              'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
#              'Loan_Amount_Term', 'Credit_History', 'Property_Area')

input_data = ('Male', 'No', 0, 'Graduate', 'No', 5849, 0, ' ', 360, 1, 'Urban')

# Convert input data to a numpy array
input_as_np_array = np.asarray(input_data)

# Reshape data to predict for one instance
input_reshaped = input_as_np_array.reshape(1, -1)

# Define the feature names
columns = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Property_Area'
]

# Create a DataFrame with the input data
input_df = pd.DataFrame(input_reshaped, columns=columns)

# Replace categorical values with numerical equivalents
input_df.replace({
    'Married': {'No': 0, 'Yes': 1}, 
    'Gender': {'Female': 0, 'Male': 1}, 
    'Education': {'Not Graduate': 0, 'Graduate': 1}, 
    'Self_Employed': {'No': 0, 'Yes': 1}, 
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
}, inplace=True)

# Handle missing or blank values
processed_input = input_df.replace(to_replace=' ', value=0)

# Make a prediction using the trained model
prediction = model.predict(processed_input)

# Display the result
if prediction == 1:
    print('Loan Approved!')
else:
    print('Loan not Approved.')
```
