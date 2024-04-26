import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

st.set_page_config(
    # Bikin judul title page
    page_title= 'Sentiment Analysis'
)

def run():
        # Membuat Judul
    st.title('Sentiment Analysis')

    # Membuat subheader
    st.subheader('Homepage')

    # Bikin dekripsi
    st.write('**Graded Challenge 7**')

    st.write('Nama  : Kelvin Rizky Novsa')
    st.write('Batch : RMT - 028')
    st.markdown('---')

    # Tambahkan gambar
    image = Image.open('download.jpg')
    st.image(image)

    # Bikin dekripsi
    st.write('**Problem Statement**')

    # Bikin dekripsi
    st.write('McDonald, sebagai salah satu merek restoran cepat saji terbesar di dunia, ingin memahami sentimen konsumen terhadap pengalaman mereka di berbagai lokasi restoran McDonald. Melalui analisis sentimen dari review konsumen, McDonald dapat mengidentifikasi area-area yang memerlukan perbaikan, memperkuat area-area yang disukai, serta meningkatkan kepuasan pelanggan secara keseluruhan.')

    # Bikin dekripsi
    st.write('Oleh karena itu, objektif dalam program ini adalah untuk melakukan prediksi sentimen, apakah konsumen puas dengan layanan McDonald.'
             )
    st.markdown('---')

    # Bikin dekripsi
    st.write('**About Data**')

    st.write("Dataset diperoleh dari platform Kaggle dengan judul McDonald's Store Reviews yang dapat diakses melalui tombol dibawah ini :")

    st.link_button("Go to dataset", "https://www.kaggle.com/datasets/nelgiriyewithana/mcdonalds-store-reviews")

    # Bikin dekripsi
    st.write('Dibawah ini merupakan informasi dataset, yang terdiri dari 33396 baris dan 10 kolom dengan schema data sebagai berikut :')

    # show dataframe
    df = pd.read_csv("McDonald_s_Reviews.csv", encoding='latin1')

    st.dataframe(df)

if __name__ == '__main__':
    run()

