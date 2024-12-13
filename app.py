from flask import Flask, request, render_template, jsonify, session
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from flask_session import Session

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Ganti dengan kunci rahasia Anda

# Konfigurasi session
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

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

@app.route('/')
def home():
    review = session.get('review', '')
    prediction_text = session.get('prediction_text', '')
    prediction_prob = session.get('prediction_prob', '')
    return render_template('index.html', review_text=review, prediction_text=prediction_text, prediction_prob=prediction_prob)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        review_tfidf = vectorizer.transform([review])
        prediction = model.predict(review_tfidf)[0]
        prediction_prob = model.predict_proba(review_tfidf)[0]
        
        sentiment = label_encoder.inverse_transform([prediction])[0]
        prob = prediction_prob[prediction]
        
        # Simpan hasil prediksi ke dalam session
        session['review'] = review
        session['prediction_text'] = sentiment
        session['prediction_prob'] = f'{prob:.6f}'
        
        return render_template('index.html', prediction_text=sentiment, prediction_prob=f'{prob:.6f}', review_text=review)

if __name__ == '__main__':
    app.run(debug=True)
