import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('Position_Salaries.csv')

X = df[['Level']]
Y = df['Salary']

degree = 4  
poly_features = PolynomialFeatures(degree=degree)

X_poly = poly_features.fit_transform(X)  

# Fit the model
poly_model = LinearRegression()
poly_model.fit(X_poly, Y)

# Streamlit app interface
st.title('Career Level vs Salary Prediction')

# Take user input for career level (no upper limit)
career_level = st.number_input('Enter Career Level:', min_value=1)

# Ensure the input is valid
if career_level:
    cl_poly = poly_features.transform([[career_level]])  
    salary = poly_model.predict(cl_poly)  # Predict salary
    
    st.write(f"Predicted Salary : {salary[0]:.2f}")
