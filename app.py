import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import pytz
from datetime import datetime
from matplotlib.ticker import MultipleLocator, FuncFormatter

# === Load model dan komponen ===
model = joblib.load('GradientBoostingClassifier - Simpia Learn Piano Fast.pkl')
vectorizer = joblib.load('tfidf_vectorizer_Simpia Learn Piano Fast.pkl')
label_encoder = joblib.load('label_encoder_Simpia Learn Piano Fast.pkl')

# === Label & Warna ===
label_map = {'positive': 'Positif', 'negative': 'Negatif'}
color_map = {'Positif': 'blue', 'Negatif': 'red'}

# === Judul Aplikasi ===
st.set_page_config(page_title="🎹 Sentiment App – Simpia Piano", layout="centered")
st.title("🎹 Aplikasi Analisis Sentimen – Simpia Learn Piano Fast")

# === Pilih Mode Input ===
st.header("\ud83d\udccc Pilih Metode Input")
input_mode = st.radio("Pilih salah satu:", ["\ud83d\udcdd Input Manual", "\ud83d\udcc1 Upload File CSV"])

# === Zona waktu WIB ===
wib = pytz.timezone("Asia/Jakarta")
now_wib = datetime.now(wib)

# ========================================
# Fungsi Visualisasi Bar Chart
# ========================================
def tampilkan_bar_chart(filtered_df):
    st.subheader("\ud83d\udcca Distribusi Sentimen – Diagram Batang")
    sentimen_bahasa = filtered_df['predicted_sentiment'].map(label_map)
    bar_data = sentimen_bahasa.value_counts().reset_index()
    bar_data.columns = ['Sentimen', 'Jumlah']
    colors = [color_map.get(sent, 'gray') for sent in bar_data['Sentimen']]

    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
    bars = ax_bar.bar(bar_data['Sentimen'], bar_data['Jumlah'], color=colors)

    for bar in bars:
        height = bar.get_height()
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            height * 0.5,
            f'{height:,.0f}'.replace(',', '.'),
            ha='center', va='center', fontsize=10, color='white', fontweight='bold'
        )

    ax_bar.yaxis.set_major_locator(MultipleLocator(50))
    ax_bar.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'.replace(',', '.')))
    max_count = bar_data['Jumlah'].max()
    ax_bar.set_ylim(0, ((max_count // 50) + 1) * 50)

    ax_bar.set_ylabel("Jumlah")
    ax_bar.set_xlabel("Sentimen")
    ax_bar.set_title("Distribusi Sentimen Pengguna – Simpia Learn Piano Fast")
    st.pyplot(fig_bar)

# === Placeholder untuk implementasi lainnya seperti input dan file upload ===
# Gunakan fungsi tampilkan_bar_chart(filtered_df) ketika ingin menampilkan diagram batang
