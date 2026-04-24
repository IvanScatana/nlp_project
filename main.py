import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st

st.set_page_config(
    page_title="Эволюция NLP",
    page_icon="🧠",
    layout="wide",
)

# --- Страницы ---
home_page = st.Page(
    "pages/0_Главная страница.py",
    title="Обзор проекта",
    icon="🏠"
)

sentiment_page = st.Page(
    "pages/1_Классификация отзывов.py",
    title="Сентимент-анализ отзывов",
    icon="📊"
)

sentiment_metrics_page = st.Page(
    "pages/2_Метрики моделей.py",
    title="Метрики (отзывы)",
    icon="📰"
)

news_page = st.Page(
    "pages/3_Классификация тематики новостей Telegram.py",
    title="Классификация новостей",
    icon="🤖"
)

news_metrics_page = st.Page(
    "pages/4_Метрики моделей2.py",
    title="Метрики (новости)",
    icon="📈"
)

llm_lora_page = st.Page(
    "pages/5_Генерация текста LLM и LoRA.py",
    title="LLM & LoRA",
    icon="📊"
)

# --- Группировка: главная страница без раздела ---
pages = {
    "": [home_page],                    # ← пустая строка = нет заголовка раздела
    "📋 Отзывы пациентов": [
        sentiment_page,
        sentiment_metrics_page,
    ],
    "📰 Новости Telegram": [
        news_page,
        news_metrics_page,
    ],
    "🤖 Генерация текста": [
        llm_lora_page,
    ],
}

pg = st.navigation(pages)
pg.run()