'''
@devolped by: **Mukesh Kanna**
'''

# -*- coding: utf-8 -*-
'''
@devolped by: **Mukesh Kanna**
'''
import pickle
import streamlit as st
import pandas as pd
import pandas as pd

model_file = 'data/model.bin'



with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


def main():
	st.title("CUSTOMER CHURN PREDICTION")
	gender = st.selectbox('Gender:', ['male', 'female'])
	seniorcitizen= st.selectbox('Does customer is a senior citizen:', [0, 1])
	partner= st.selectbox(' Does customer has a partner:', ['yes', 'no'])
	dependents = st.selectbox(' Does customer has  dependents:', ['yes', 'no'])
	phoneservice = st.selectbox('Does customer has phoneservice:', ['yes', 'no'])
	multiplelines = st.selectbox('Does customer has multiplelines:', ['yes', 'no', 'no_phone_service'])
	internetservice= st.selectbox('Does customer has internetservice:', ['dsl', 'no', 'fiber_optic'])
	onlinesecurity= st.selectbox(' Does customer has onlinesecurity:', ['yes', 'no', 'no_internet_service'])
	onlinebackup = st.selectbox(' Does customer has onlinebackup:', ['yes', 'no', 'no_internet_service'])
	deviceprotection = st.selectbox(' Does customer has deviceprotection:', ['yes', 'no', 'no_internet_service'])
	techsupport = st.selectbox(' Does customer has techsupport:', ['yes', 'no', 'no_internet_service'])
	streamingtv = st.selectbox(' Does customer has streamingtv:', ['yes', 'no', 'no_internet_service'])
	streamingmovies = st.selectbox(' Does customer has streamingmovies:', ['yes', 'no', 'no_internet_service'])
	contract= st.selectbox(' Does customer has a contract:', ['month-to-month', 'one_year', 'two_year'])
	paperlessbilling = st.selectbox(' Does customer has a paperlessbilling:', ['yes', 'no'])
	paymentmethod= st.selectbox('Payment Option:', ['bank_transfer_(automatic)', 'credit_card_(automatic)', 'electronic_check' ,'mailed_check'])
	tenure = st.number_input('No of months the customer been with  current telco provider :', min_value=0, max_value=240, value=0)
	monthlycharges= st.number_input('Monthly charges :', min_value=0, max_value=240, value=0)
	totalcharges = tenure*monthlycharges
	output= ""
	input_dict={
			"gender":gender ,
			"seniorcitizen": seniorcitizen,
			"partner": partner,
			"dependents": dependents,
			"phoneservice": phoneservice,
			"multiplelines": multiplelines,
			"internetservice": internetservice,
			"onlinesecurity": onlinesecurity,
			"onlinebackup": onlinebackup,
			"deviceprotection": deviceprotection,
			"techsupport": techsupport,
			"streamingtv": streamingtv,
			"streamingmovies": streamingmovies,
			"contract": contract,
			"paperlessbilling": paperlessbilling,
			"paymentmethod": paymentmethod,
			"tenure": tenure,
			"monthlycharges": monthlycharges,
			"totalcharges": totalcharges
			}

	if st.button("Predict"):
					
		X = dv.transform([input_dict])
		y_pred = model.predict_proba(X)[0, 1]
		churn = y_pred >= 0.5
		output = bool(churn)
		st.success('Customer Churn status  : {0} '.format(output))
				
	
	
if __name__ == '__main__':
	main()



'''
**© 2023 Customer churn predictions. All rights reserved.**
'''
