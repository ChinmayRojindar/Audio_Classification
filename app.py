import streamlit as st
from Dataset import dataset_information_page
from Predict_page import audio_classification_page



def main():
    
    st.title('Audio classification app')

    menu=['Dataset','Model']
    choice=st.sidebar.selectbox('Select Page',menu)

    if choice=='Model':
        audio_classification_page()

    elif choice=="Dataset":
        dataset_information_page()


if __name__ == '__main__':
    main()