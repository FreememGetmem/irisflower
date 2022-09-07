import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from PIL import Image

image_versicolor = Image.open('img/versicolor.jpg')
image_virginica = Image.open('img/virginica.jpg')
image_setosa = Image.open('img/setosa.jpg')

st.write('''
# Simple App for Iris flower Predict !!
This Application help to predict Iris flower...
''')

st.sidebar.header('Input parameters :')

def user_input():
    sepal_length = st.sidebar.slider('La longueur du Sépal :',4.3, 7.9,5.3)
    sepal_width = st.sidebar.slider('La largeur du Sépal :',2.0, 4.4,3.3)
    petal_length = st.sidebar.slider('La longueur du Pétal :',1.0, 6.9,2.3)
    petal_width = st.sidebar.slider('La Largeur du Pépal :',0.1, 2.5,1.3)

    data = {
        'sepal_length' : sepal_length,
        'sepal_width' : sepal_width,
        'petal_length' : petal_length,
        'petal_width' : petal_width
    }

    fl_parameters = pd.DataFrame(data, index=[0])
    return fl_parameters

df = user_input()
st.subheader('Let\'s predict the Iris flower category with the below parameter :')
st.write(df)

iris = datasets.load_iris()

clf = GaussianNB()
clf.fit(iris.data, iris.target)

prediction = clf.predict(df)

st.subheader('The category of the  Iris flower is :')
st.write(iris.target_names[prediction])
if (prediction==1):
    st.image(image_versicolor, use_column_width=False)
elif (prediction==0):
    st.image(image_setosa, use_column_width=False)
else :
    st.image(image_virginica, use_column_width=False)
