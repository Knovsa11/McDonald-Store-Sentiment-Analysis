import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import plotly.express as px
from PIL import Image
from wordcloud import WordCloud

st.set_page_config(
    # Bikin judul title page
    page_title= 'Sentimen Analysis'
)

def run():
    # Membuat Judul
    st.title('Sentimen Analysis')

    # Membuat subheader
    st.subheader('Exploratory Data Analysis (EDA)')

    # Bikin dekripsi
    st.write('**Distribusi Data Rating**')

    # Bikin dekripsi
    st.write('Persebaran rating didominasi oleh rating 5, yang berarti konsumen puas dengan pelayanan McDonald')

    # show dataframe
    df = pd.read_csv("McDonald_s_Reviews.csv", encoding='latin1')

    df = df.drop('reviewer_id', axis=1)
    # Menghitung jumlah data pada setiap kategori rating
    rating_distribution = df.groupby('rating').size()

    # Membuat pie chart berdasarkan distribusi rating
    plt.figure()
    plt.pie(rating_distribution, labels=rating_distribution.index, autopct='%1.1f%%', colors=sns.color_palette('Dark2', len(rating_distribution)))
    plt.title('Distribusi Rating')
    plt.axis('equal')

    # Menyimpan plot sebagai objek
    pie_chart_fig = plt.gcf()

    # Menampilkan plot menggunakan st.pyplot()
    st.pyplot(pie_chart_fig)
    st.markdown('---')

    # Bikin dekripsi
    st.write('**Distribusi Kata**')

    # Bikin dekripsi
    st.write('Berdasarkan visualisasi diatas, jumlah kata berkumpul pada rentang jumlah kata tertentu sehingga distribusi data tidak terdistribusi normal..')

    # Menambah kolom total word untuk melihat jumlah kata dalam sebuah review
    df['total_words'] = df['review'].apply(lambda x: len(nltk.word_tokenize(x)))

    # plot distribution of total words
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='total_words', bins=20)
    plt.title('Distribusi Jumlah Kata')
    plt.xlabel('Total Words')
    plt.ylabel('Frequency')

    # Menyimpan plot sebagai objek
    histogram_fig = plt.gcf()

    # Menampilkan plot menggunakan st.pyplot()
    st.pyplot(histogram_fig)
    st.markdown('---')

    # Bikin dekripsi
    st.write('**WordCloud**')

    # Bikin dekripsi
    st.write('Berdasarkan visualisasi didapatkan kata yang paling banyak muncul ditiap rating. Kemudian diperoleh insight juga bahwa perlu dilakukan stopword untuk dapat menghasilkan prediksi yang baik')

    # List of rating categories
    rating_categories = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']

    # Iterate through each rating category
    for rating_category in rating_categories:
        # Select data for each rating
        text_combined = ' '.join(df[df['rating'] == rating_category]['review'])

        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text_combined)

        # Display word cloud
        wordcloud_fig = plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud for {rating_category}')
        plt.axis('off')
        st.pyplot(wordcloud_fig)

if __name__ == '__main__':
    run()


