import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv('Position_Salaries.csv')

X = df[['Level']]
Y = df['Salary']

# Polynomial Regression Setup
degree = 4  # Degree of the polynomial
poly_features = PolynomialFeatures(degree=degree)

X_poly = poly_features.fit_transform(X)  # Transform X to include polynomial features

# Fit the model
poly_model = LinearRegression()
poly_model.fit(X_poly, Y)

# Create a title for the app
st.title("Career Level vs Salary Prediction")
st.write("This app predicts the salary for a given career level based on a polynomial regression model.")

# Create an input box for users to enter career level
career_level = st.number_input("Enter Career Level:", min_value=1, max_value=10, value=1)

# Prediction for the user input
cl_poly = poly_features.transform([[career_level]])  # Use .transform(), NOT .fit_transform()
salary = poly_model.predict(cl_poly)  # Predict salary

# Display the prediction
st.write(f"Predicted Salary for Career Level {career_level}: ${salary[0]:,.2f}")

# Visualization of the polynomial regression
st.write("### Polynomial Regression Visualization")
fig, ax = plt.subplots()
ax.scatter(X, Y, color='red')
ax.plot(X, poly_model.predict(X_poly), color='green')
ax.set_title('Career Level vs Salary')
ax.set_xlabel('Career Level')
ax.set_ylabel('Salary')
st.pyplot(fig)

# R2 Score
poly_predict = poly_model.predict(X_poly)
r2_poly = r2_score(Y, poly_predict)
st.write(f"R-squared for Polynomial Regression: {r2_poly:.4f}")
