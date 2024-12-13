import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler

# Fungsi untuk memuat data dan melatih model
def train_model():
    dataset = pd.read_csv('model data ulasan hp oppo.csv')
    X = dataset['final_Ulasan']
    y = dataset['Sentiment']

    # Mengonversi label sentimen menjadi numerik
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(X)

    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_tfidf, y_encoded)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_resampled, y_resampled)
    
    return knn, vectorizer, label_encoder

# Muat model, vectorizer, dan label encoder
model, vectorizer, label_encoder = train_model()

# Memuat file CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Aplikasi Streamlit
st.title("Sentiment Analysis")

# Memuat CSS
load_css('style.css')

# Input ulasan dari pengguna
review = st.text_area("Tulis ulasan Anda di sini...")

if st.button("Prediksi Sentimen"):
    if review:
        review_tfidf = vectorizer.transform([review])
        prediction = model.predict(review_tfidf)[0]
        prediction_prob = model.predict_proba(review_tfidf)[0]
        
        sentiment = label_encoder.inverse_transform([prediction])[0]
        prob = prediction_prob[prediction]
        
        st.write(f"**Sentimen**: {sentiment}")
        st.write(f"**Probabilitas**: {prob:.6f}")
        st.write(f"**Ulasan**: {review}")
        
        # Menampilkan ikon emosi
        if sentiment == "Negatif":
            st.image("Negatif.jpg", width=50)
        elif sentiment == "Netral":
            st.image("Netral.jpg", width=50)
        elif sentiment == "Positif":
            st.image("Positif.jpg", width=50)
    else:
        st.write("Silakan masukkan ulasan terlebih dahulu.")
