# import libraries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

from PIL import Image

# title
st.write(""" # Iris Flower Prediction App
This app predicts the Iris flower type based on User input parameters.
""")

# information abaut df set
st.info("Iris dataset is taken from Sir R.A. Fisher paper for pattern recognition literature. It is also known as Anderson’s Iris data set as Edge Anderson originally collected the data to quantify the variation of Iris flowers of there different class. These class are class Iris-Setosa, Iris-Versicolour, Iris-Virginica with attributes as Sepal Length, Sepal Width, Petal Length and Petal Width in centimeters.")

# display plant image
img_iris = Image.open("iris_type.png")
st.image(img_iris)


#_______________________________IMPUT PARAMETERS
# create input parameters
st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, .5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

# data from User in sidebar
df = user_input_features()

# table with input parameters
st.subheader('User Input parameters:')
st.write(df)



#________________________________PREDICTION
#load iris data
iris = datasets.load_iris()
X = iris.data
Y = iris.target # class 0/1/2

# Random Frorest Classifier to predict
clf = RandomForestClassifier()
clf.fit(X, Y)

# Flower prediction and probability based on User parameters 
prediction = clf.predict(df)
prediction_prob = clf.predict_proba(df)

# class labels
st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

# flower prediction
st.subheader('Prediction')
st.write(iris.target_names[prediction])

# display plant prediction image
if(prediction == 0):
    setosa_0 = Image.open("setosa.png")
    st.image(setosa_0)
elif(prediction == 1):
    versicolor_1 = Image.open("versicolor.png")
    st.image(versicolor_1)
else:
    virginica_2 = Image.open("virginica.png")
    st.image(virginica_2)

# prediction probability
st.subheader('Prediction Probability')
st.write(prediction_prob)