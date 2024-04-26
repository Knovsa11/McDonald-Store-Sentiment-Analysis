import streamlit as st
import Homepage
import eda
import prediction 

page = st.sidebar.selectbox('Pilih Halaman: ', ('Homepage','EDA', 'Prediction'))

if page == "Homepage":
    Homepage.run()
elif page == "EDA":
    eda.run()
else:
    prediction.run()


