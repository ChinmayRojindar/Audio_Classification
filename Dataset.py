import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def dataset_information_page():
    st.header('Dataset information page')
    

    metadata_path="UrbanSound8K\\metadata\\UrbanSound8K.csv"
    df=pd.read_csv(metadata_path)
    #print(df.head())

    #Display basic dataset info
    st.write("Number of samples:",len(df))   
    st.write("Number of classes:",len(df['class'].unique()))

    #Data for pie chart
    class_count=df['class'].value_counts()
    class_labels=class_count.index.to_list()
    class_values=class_count.to_list()

    #create a pie chart
    fig,ax=plt.subplots()
    ax.pie(class_values,labels=class_labels,autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    st.write("Dataset link : https://urbansounddataset.weebly.com/download-urbansound8k.html")