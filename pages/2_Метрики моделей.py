# pages/2_Визуализация_метрик.py
import streamlit as st
import pandas as pd
from PIL import Image
import os

# Путь к папке с метриками
BASE_DIR = "Metrics/Reviews/"

st.set_page_config(
    page_title="Визуализация метрик",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Визуализация метрик моделей")

def load_image(relative_path):
    """Загружает изображение из BASE_DIR."""
    full_path = os.path.join(BASE_DIR, relative_path)
    if os.path.exists(full_path):
        return Image.open(full_path)
    else:
        st.warning(f"Изображение не найдено: {full_path}")
        return None

# 1. Детальная таблица метрик
st.header("📋 Детальная таблица метрик")
csv_path = os.path.join(BASE_DIR, "model_metrics_all.csv")
try:
    metrics_df = pd.read_csv(csv_path)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    csv = metrics_df.to_csv(index=False)
    st.download_button(
        label="📥 Скачать CSV",
        data=csv,
        file_name="model_metrics.csv",
        mime="text/csv"
    )
except FileNotFoundError:
    st.warning(f"Файл {csv_path} не найден. Использую встроенные данные.")
    demo = pd.DataFrame({
        "Модель": ["Logistic Regression", "Random Forest", "LSTM", "ImprovedTinyBERTFull"],
        "Accuracy": [0.9503, 0.8885, 0.9250, 0.9101],
        "F1-macro": [0.9489, 0.8868, 0.9226, 0.9073],
    })
    st.dataframe(demo, use_container_width=True, hide_index=True)

st.divider()

# 2. Сравнение F1-macro
st.header("🎯 Сравнение F1-macro")
img = load_image("f1_macro_comparison.png")
if img:
    st.image(img, use_container_width=True)

st.divider()

# 3. Матрицы ошибок
st.header("🔢 Матрицы ошибок")
img = load_image("confusion_matrices_all.png")
if img:
    st.image(img, use_container_width=True)

st.divider()

# 4. Важность признаков
st.header("🔍 Важность признаков")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Logistic Regression – Top 20")
    img = load_image("logreg_top20_lollipop.png")
    if img:
        st.image(img, use_container_width=True)
with col2:
    st.subheader("Random Forest")
    img = load_image("rf_importances.png")
    if img:
        st.image(img, use_container_width=True)

st.divider()

# 5. Кривые обучения
st.header("📈 Кривые обучения")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Logistic Regression")
    img = load_image("learning_curve_Logistic Regression.png")
    if img:
        st.image(img, use_container_width=True)
with col2:
    st.subheader("Random Forest")
    img = load_image("learning_curve_Random Forest.png")
    if img:
        st.image(img, use_container_width=True)

st.subheader("LSTM")
img = load_image("lstm_learning_curve.png")
if img:
    st.image(img, use_container_width=True)

st.subheader("ImprovedTinyBERT")
img = load_image("bert_training_curves.png")
if img:
    st.image(img, use_container_width=True)