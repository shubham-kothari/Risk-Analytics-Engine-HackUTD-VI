# HackUTD-VI
### MORE DATA AT https://drive.google.com/open?id=1HTGBKv7wGN-wTXFfzBQWeAqq7P8wDRXg

### LINK TO TABLEAU WORKBOOK https://public.tableau.com/profile/shubham.kothari#!/vizhome/FannieMae-MortgageAnalysis/Story1?publish=yes


## Inspiration
As a part of Fanni Mae's HackUTD-VI Data Science Challenge, we built a Risk Analytics Engine using Machine Learning and Data Science. This is a hosted Web-Application that is based on a highly-accurate ML Model to Predict if a certain loan should be acquired or not.

## How we built it
The dataset contained 3rd Quarter Single-Family Loan Acquisition and Performane data for Years 2004, 2008, 2012, and 2016. In this scenario, our main focus was to isolate Pre-Financial-Crisis and Post-Financial-Crisis data so that our Analysis and Predictive model is not biased towards certain trends.

After carefully studying the Glossary, we identified 3 most-important Risk-Factors that help in identifying the Default cases. Delinquency Rate, Zero Balance Code, and Foreclosure were carefully analyzed and transformed to form a single Target Variable that can acurately identify Risky Loans.

By using the Data Analysis notebook provided by FannieMae as an inspiration, we cleaned the data and transformed it into a subset of most-important features (Identified by Analysis and Ensemble ML models).
We built multiple ML models and based on the **Ease of Understanding and Prediction Accuracy** we selected a Random Forest Classifier that has an **Average Precision-Recall Score: 98.59%** and **Area Under ROC: 98.63%**.

### Note: All Results are based on a 10-Fold Cross-Validation

## End Results

1. A Web Application based User-Interface that can be used to see the Analysis, Resulst/Model Performance, and Predict Loan Acquisition Risk.
<br><br>
2. A **Predictive Model Markup Language (PMML)** based Machine Learning Model that is **Ready for Production Deployment** irrespective of the platform.
<br><br>
3. A Taleau Workbook that contains **Data Analysis Dashboard** to better understand the Data.
