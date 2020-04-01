import streamlit as st
import pandas as pd
import pickle as pkl
import numpy as np
from PIL import Image

# Reuse this data across runs!
st.title("Risk Analytics")

st.sidebar.title("Choose what you want to see")
page = st.sidebar.selectbox('',options=['Select Option', 'Analysis','Results','Predict Something'])

if page == 'Select Option':
	st.image("FannieMae.png")
	
	st.markdown("""## Inspiration
As a part of Fanni Mae's HackUTD-VI Data Science Challenge, we built a Risk Analytics Engine using Machine Learning and Data Science. This is a hosted Web-Application that is based on a highly-accurate ML Model to Predict if a certain loan should be acquired or not.""")

	st.markdown("""## How we built it
The dataset contained 3rd Quarter Single-Family Loan Acquisition and Performane data for Years 2004, 2008, 2012, and 2016. In this scenario, our main focus was to isolate Pre-Financial-Crisis and Post-Financial-Crisis data so that our Analysis and Predictive model is not biased towards certain trends.

After carefully studying the Glossary, we identified 3 most-important Risk-Factors that help in identifying the Default cases. Delinquency Rate, Zero Balance Code, and Foreclosure were carefully analyzed and transformed to form a single Target Variable that can acurately identify Risky Loans.

By using the Data Analysis notebook provided by FannieMae as an inspiration, we cleaned the data and transformed it into a subset of most-important features (Identified by Analysis and Ensemble ML models).
We built multiple ML models and based on the **Ease of Understanding and Prediction Accuracy** we selected a Random Forest Classifier that has an **Average Precision-Recall Score: 98.59%** and **Area Under ROC: 98.63%**.

### Note: All Results are based on a 10-Fold Cross-Validation""")

	st.markdown("""## End Results

1. A Web Application based User-Interface that can be used to see the Analysis, Resulst/Model Performance, and Predict Loan Acquisition Risk.

2. A **Predictive Model Markup Language (PMML)** based Machine Learning Model that is **Ready for Production Deployment** irrespective of the platform.

3. A Taleau Workbook that contains **Data Analysis Dashboard** to better understand the Data.""")


if page == 'Predict Something':

	st.sidebar.title("Interactive Risk Analytics Engine")
	st.sidebar.markdown(
    	"""
	Confused if you should accept or reject the new loan application?
	Try our Risk Analyzer and improve your profits by avoiding risky acquisitions!
	"""
	)

	options = {'Seller': ['CITIMORTGAGE, INC.', 'BANK OF AMERICA, N.A.', 'OTHER',
       'SUNTRUST MORTGAGE INC.', 'WELLS FARGO BANK, N.A.',
       'FIRST TENNESSEE BANK NATIONAL ASSOCIATION',
       'HARWOOD STREET FUNDING I, LLC', 'GMAC MORTGAGE, LLC',
       'JPMORGAN CHASE BANK, NA', 'REGIONS BANK',
       'PRINCIPAL RESIDENTIAL MORTGAGE CAPITAL RESOURCES, LLC',
       'FLAGSTAR BANK, FSB', 'AMTRUST BANK',
       'BISHOPS GATE RESIDENTIAL MORTGAGE TRUST',
       'IRWIN MORTGAGE, CORPORATION', 'USAA FEDERAL SAVINGS BANK',
       'RBC MORTGAGE COMPANY', 'PNC BANK, N.A.',
       'JPMORGAN CHASE BANK, NATIONAL ASSOCIATION', 'CHASE HOME FINANCE',
       'HSBC BANK USA, NATIONAL ASSOCIATION', 'CHASE HOME FINANCE, LLC',
       'FLAGSTAR CAPITAL MARKETS CORPORATION', 'PHH MORTGAGE CORPORATION',
       'THE BRANCH BANKING AND TRUST COMPANY',
       'FDIC, RECEIVER, INDYMAC FEDERAL BANK FSB', 'ALLY BANK',
       'QUICKEN LOANS INC.', 'STEARNS LENDING, LLC',
       'PROVIDENT FUNDING ASSOCIATES, L.P.', 'CASHCALL, INC.',
       'FRANKLIN AMERICAN MORTGAGE COMPANY',
       'AMERISAVE MORTGAGE CORPORATION', 'PENNYMAC CORP.',
       'NYCB MORTGAGE COMPANY, LLC', 'FEDERAL HOME LOAN BANK OF CHICAGO',
       'HOMEWARD RESIDENTIAL, INC.',
       'CHICAGO MORTGAGE SOLUTIONS DBA INTERFIRST MORTGAGE COMPANY',
       'UNITED SHORE FINANCIAL SERVICES, LLC D/B/A UNITED WHOLESALE MORTGAGE',
       'IMPAC MORTGAGE CORP.', 'SUNTRUST BANK',
       'FINANCE OF AMERICA MORTGAGE LLC', 'U.S. BANK N.A.',
       'PMT CREDIT RISK TRANSFER TRUST 2016-1', 'FREEDOM MORTGAGE CORP.',
       'MOVEMENT MORTGAGE, LLC', 'NATIONSTAR MORTGAGE, LLC',
       'LOANDEPOT.COM, LLC', 'CALIBER HOME LOANS, INC.'],
	'Channel':['R', 'C', 'B'],
	'Origin_Year': [2004,2008,2012,2016],
	'First_Time_Buyer': ['N', 'Y', 'U'],
	'Loan_Purpose': ['R', 'C', 'P', 'U'],
	'FICO_bins': ['660-700', '740-780', '700-740', '780+', '620-660', '0-620'],
	'Term_bins': ['<= 30 Years', '<=15 Years']
	}

	train_cols = ['Origin_Year', 'Interest_Rate', 'UPB', 'LTV', 'Num_Borrowers',
       'LOAN AGE', 'Channel_B', 'Channel_C', 'Channel_R', 'Seller_ALLY BANK',
       'Seller_AMERISAVE MORTGAGE CORPORATION', 'Seller_AMTRUST BANK',
       'Seller_BANK OF AMERICA, N.A.',
       'Seller_BISHOPS GATE RESIDENTIAL MORTGAGE TRUST',
       'Seller_CALIBER HOME LOANS, INC.', 'Seller_CASHCALL, INC.',
       'Seller_CHASE HOME FINANCE', 'Seller_CHASE HOME FINANCE, LLC',
       'Seller_CHICAGO MORTGAGE SOLUTIONS DBA INTERFIRST MORTGAGE COMPANY',
       'Seller_CITIMORTGAGE, INC.',
       'Seller_FDIC, RECEIVER, INDYMAC FEDERAL BANK FSB',
       'Seller_FEDERAL HOME LOAN BANK OF CHICAGO',
       'Seller_FINANCE OF AMERICA MORTGAGE LLC',
       'Seller_FIRST TENNESSEE BANK NATIONAL ASSOCIATION',
       'Seller_FLAGSTAR BANK, FSB',
       'Seller_FLAGSTAR CAPITAL MARKETS CORPORATION',
       'Seller_FRANKLIN AMERICAN MORTGAGE COMPANY',
       'Seller_FREEDOM MORTGAGE CORP.', 'Seller_GMAC MORTGAGE, LLC',
       'Seller_HARWOOD STREET FUNDING I, LLC',
       'Seller_HOMEWARD RESIDENTIAL, INC.',
       'Seller_HSBC BANK USA, NATIONAL ASSOCIATION',
       'Seller_IMPAC MORTGAGE CORP.', 'Seller_IRWIN MORTGAGE, CORPORATION',
       'Seller_JPMORGAN CHASE BANK, NA',
       'Seller_JPMORGAN CHASE BANK, NATIONAL ASSOCIATION',
       'Seller_LOANDEPOT.COM, LLC', 'Seller_MOVEMENT MORTGAGE, LLC',
       'Seller_NATIONSTAR MORTGAGE, LLC', 'Seller_NYCB MORTGAGE COMPANY, LLC',
       'Seller_OTHER', 'Seller_PENNYMAC CORP.',
       'Seller_PHH MORTGAGE CORPORATION',
       'Seller_PMT CREDIT RISK TRANSFER TRUST 2016-1', 'Seller_PNC BANK, N.A.',
       'Seller_PRINCIPAL RESIDENTIAL MORTGAGE CAPITAL RESOURCES, LLC',
       'Seller_PROVIDENT FUNDING ASSOCIATES, L.P.',
       'Seller_QUICKEN LOANS INC.', 'Seller_RBC MORTGAGE COMPANY',
       'Seller_REGIONS BANK', 'Seller_STEARNS LENDING, LLC',
       'Seller_SUNTRUST BANK', 'Seller_SUNTRUST MORTGAGE INC.',
       'Seller_THE BRANCH BANKING AND TRUST COMPANY', 'Seller_U.S. BANK N.A.',
       'Seller_UNITED SHORE FINANCIAL SERVICES, LLC D/B/A UNITED WHOLESALE MORTGAGE',
       'Seller_USAA FEDERAL SAVINGS BANK', 'Seller_WELLS FARGO BANK, N.A.',
       'First_Time_Buyer_N', 'First_Time_Buyer_U', 'First_Time_Buyer_Y',
       'Loan_Purpose_C', 'Loan_Purpose_P', 'Loan_Purpose_R', 'Loan_Purpose_U',
       'FICO_bins_0-620', 'FICO_bins_620-660', 'FICO_bins_660-700',
       'FICO_bins_700-740', 'FICO_bins_740-780', 'FICO_bins_780+',
       'Term_bins_<= 30 Years', 'Term_bins_<=15 Years']

	data = {'Channel':'R',
 'Origin_Year':2004,
 'Seller':'FIRST TENNESSEE BANK NATIONAL ASSOCIATION',
 'Interest_Rate':4.125,
 'UPB':176000,
 'LTV':75,
 'Num_Borrowers':2.0,
 'First_Time_Buyer':'N',
 'Loan_Purpose':'P',
 'LOAN AGE':34,
 'FICO_bins':'660-700',
 'Term_bins':'<=30 Years'}

	dummy_cols = ['Channel',
 'Seller',
 'First_Time_Buyer',
 'Loan_Purpose',
 'FICO_bins',
 'Term_bins']
	value_cols = ['LOAN AGE', 'Interest_Rate', 'UPB', 'LTV', 'Num_Borrowers']

	def preprocess(idata):
		dic = {}
		for col in train_cols:
			if col in dummy_cols:
				dic[col+'_'+idata[col]] = 1
			elif col in value_cols:
				dic[col] = idata[col]
			else:
				dic[col] = 0
		dic['UPB'] = np.log1p(dic['UPB'])
		dic['Origin_Year'] = int(dic['Origin_Year'])
        
		df = pd.DataFrame(dic,index=[0])
    
		return df
    
	#st.sidebar.selectbox('Which model do you want to use?', options['Year'])
	input_data = {}

	for key in list(data.keys()):
		if key in dummy_cols:
			input_data[key] = st.sidebar.selectbox(key, options[key])
		if key in value_cols:
			input_data[key] = st.number_input(key,0)
		if key == 'Origin_Year':
			input_data[key] = st.radio('Choose Origin_Year',options['Origin_Year'])
		
	with open('Models/rf_model.pkl','rb') as model:
		clf = pkl.load(model)
    
	#st.header("Risk Analytics")
	if st.button("Predict!"):
		#print(input_data)
		result = clf.predict(preprocess(input_data)) 
		#print(preprocess(input_data))
		#print(result)
		if result==1:
			st.write("REJECT")
		else:
			st.write("ACCEPT")
		
if page == 'Results':
	st.write("Scores")
	st.image("Results/SCORES.png")
	
	st.write("Classification Report")
	st.image("Results/REPORT.png")
	
	st.write("ROC Curve")
	st.image('Results/ROC.png')
	
if page == 'Analysis':
	
	st.write("See Tableau Workbook at https://public.tableau.com/profile/shubham.kothari#!/vizhome/FannieMae-MortgageAnalysis/Story1?publish=yes")
	st.write("FICO Score Distribution for different Loan Status")
	st.image("Results/FICO Score Distribution for different Loan Status.png")
	
	st.write("Loan Status Over the Years")
	st.image("Results/Loan Status Over the Years.png")
	
	st.write("Loan to Value (LTV) Distribution for different Loan Status")
	st.image('Results/Loan to Value (LTV) Distribution for different Loan Status.png')
	
	
	st.write("Top 15 States based on the % of Loan under different Loan Status")
	st.image('Results/Top 15 States based on the % of Loan under different Loan Status.png')
