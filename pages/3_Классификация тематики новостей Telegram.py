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
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel
import numpy as np
import time
import joblib
import requests
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import pandas as pd

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# ------------------------------------------------------------
# 1. Утилиты загрузки файлов с Hugging Face
# ------------------------------------------------------------
BASE_URL = "https://huggingface.co/Scatana/nlp_project/resolve/main"
CACHE_DIR = Path("models_cache")
CACHE_DIR.mkdir(exist_ok=True)

@st.cache_resource
def download_file(filename):
    local_path = CACHE_DIR / filename
    if not local_path.exists():
        url = f"{BASE_URL}/{filename}"
        with st.spinner(f"Загрузка {filename}..."):
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return local_path

# ------------------------------------------------------------
# 2. Кастомная модель ImprovedTinyBERTFull (многоклассовая)
# ------------------------------------------------------------
class ImprovedTinyBERTFull(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3, hidden_dim=512):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            "cointegrated/rubert-tiny2",
            attn_implementation="eager"
        )
        for param in self.bert.parameters():
            param.requires_grad = True

        bert_dim = 312
        self.pooling_dim = bert_dim * 3

        self.bn_input = nn.BatchNorm1d(self.pooling_dim)
        self.fc1 = nn.Linear(self.pooling_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.residual_proj = nn.Linear(self.pooling_dim, hidden_dim // 2) if self.pooling_dim != hidden_dim // 2 else nn.Identity()
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        self._init_weights()

    def _init_weights(self):
        for module in [self.fc1, self.fc2, self.classifier]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, output_attentions=False):
        bert_out = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask,
                             output_attentions=True)
        last_hidden = bert_out.last_hidden_state
        attentions = bert_out.attentions

        cls_emb = last_hidden[:, 0, :]
        mean_emb = last_hidden.mean(dim=1)
        max_emb, _ = last_hidden.max(dim=1)
        pooled = torch.cat([cls_emb, mean_emb, max_emb], dim=1)
        pooled = self.bn_input(pooled)

        out = self.fc1(pooled)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        residual = self.residual_proj(pooled)
        out = torch.relu(out + residual)
        out = self.dropout2(out)
        logits = self.classifier(out)

        if output_attentions:
            return logits, attentions
        return logits

# ------------------------------------------------------------
# 3. Загрузка ресурсов (все кэшируются) – файлы с суффиксом 2
# ------------------------------------------------------------
@st.cache_resource
def load_tfidf_vectorizer():
    return joblib.load(download_file("tfidf_vectorizer2.pkl"))

@st.cache_resource
def load_logreg():
    return joblib.load(download_file("logreg_tfidf2.pkl"))

@st.cache_resource
def load_rf():
    return joblib.load(download_file("rf_tfidf2.pkl"))

@st.cache_resource
def load_lstm_tokenizer():
    return joblib.load(download_file("tokenizer_lstm2.pkl"))

@st.cache_resource
def load_lstm_model():
    path = download_file("best_lstm_model2.keras")
    model = tf.keras.models.load_model(path, compile=False)
    # Принудительная инициализация
    dummy = tf.keras.preprocessing.sequence.pad_sequences([[0]], maxlen=128)
    _ = model.predict(dummy, verbose=0)
    return model

@st.cache_resource
def load_bert():
    # Файлы токенизатора общие, не требуют суффикса 2
    download_file("tokenizer.json")
    download_file("tokenizer_config.json")
    model_pt = download_file("best_ImprovedTinyBERTFull2.pt")
    tokenizer = AutoTokenizer.from_pretrained(str(CACHE_DIR))
    # num_classes = 5 (крипта, мода, спорт, технологии, финансы)
    model = ImprovedTinyBERTFull(num_classes=5, dropout=0.3, hidden_dim=512)
    state_dict = torch.load(model_pt, map_location=torch.device('cpu'))
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return tokenizer, model

# ------------------------------------------------------------
# 4. Функции инференса и визуализации
# ------------------------------------------------------------
def predict_sklearn(model, vectorizer, text):
    t0 = time.perf_counter()
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    elapsed = (time.perf_counter() - t0) * 1000
    return pred, proba, elapsed

def predict_lstm(model, tokenizer, text, max_len=128):
    t0 = time.perf_counter()
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len, padding='post')
    proba = model.predict(padded, verbose=0)[0]
    pred = np.argmax(proba)
    elapsed = (time.perf_counter() - t0) * 1000
    return pred, proba, elapsed

def predict_bert(tokenizer, model, text):
    t0 = time.perf_counter()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits, attentions = model(input_ids=inputs['input_ids'],
                                   attention_mask=inputs['attention_mask'],
                                   output_attentions=True)
        proba = torch.softmax(logits, dim=1).numpy()[0]
        pred = np.argmax(proba)
    elapsed = (time.perf_counter() - t0) * 1000
    return pred, proba, elapsed, inputs['input_ids'][0], attentions

def highlight_attention(tokenizer, input_ids, attentions, text, layer=-1, head=0):
    """Визуализация attention: агрегация BPE-токенов в слова."""
    if attentions is None or len(attentions) == 0:
        return "<div>Карта внимания недоступна.</div>"
    try:
        encoded = tokenizer(text, return_offsets_mapping=True, truncation=True, padding=True)
        word_ids = encoded.word_ids()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        attn = attentions[layer][0, head, 0, :].detach().numpy()

        word_attn = {}
        word_token_ids = {}
        for i, tok in enumerate(tokens):
            wid = word_ids[i]
            if wid is None:
                continue
            if wid not in word_attn:
                word_attn[wid] = []
                word_token_ids[wid] = []
            word_attn[wid].append(attn[i])
            word_token_ids[wid].append(input_ids[i].item())

        words = []
        scores = []
        for wid in sorted(word_attn):
            word_str = tokenizer.decode(word_token_ids[wid], skip_special_tokens=True).strip().replace(' ', '')
            words.append(word_str)
            scores.append(np.mean(word_attn[wid]))

        scores = np.array(scores)
        norm = mcolors.Normalize(vmin=scores.min(), vmax=scores.max())
        cmap = plt.colormaps['Reds']

        html_parts = ['<div style="line-height:2.5; font-size:16px;">']
        for word, score in zip(words, scores):
            alpha = norm(score)
            r, g, b, _ = cmap(alpha)
            bg = f"rgba({int(r*255)},{int(g*255)},{int(b*255)},1.0)"
            html_parts.append(
                f'<span style="background-color:{bg}; color:black; '
                f'padding:4px 6px; margin:2px; border-radius:4px; '
                f'display:inline-block;">{word}</span>'
            )
        html_parts.append('</div>')
        return "".join(html_parts)
    except Exception as e:
        return f"<div>Ошибка визуализации: {e}</div>"

def lstm_token_importance(model, tokenizer, text, max_len=128):
    """Градиентная карта важности слов для LSTM."""
    emb_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Embedding):
            emb_layer = layer
            break
    if emb_layer is None:
        return "<div>Embedding слой не найден</div>"
    vocab_size = emb_layer.input_dim

    from tensorflow.keras.preprocessing.text import text_to_word_sequence
    raw_words = text_to_word_sequence(text)
    if not raw_words:
        return "<div>Не удалось выделить слова.</div>"

    seq = []
    for word in raw_words:
        idx = tokenizer.word_index.get(word, 0)
        if idx >= vocab_size:
            idx = 0
        seq.append(idx)
    padded = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=max_len, padding='post')
    input_tensor = tf.constant(padded)

    tail_layers = model.layers[model.layers.index(emb_layer) + 1:]
    tail_model = tf.keras.Sequential(tail_layers)

    with tf.GradientTape() as tape:
        embeddings = emb_layer(input_tensor)
        tape.watch(embeddings)
        preds = tail_model(embeddings)
        top_class = tf.argmax(preds[0])
        loss = preds[0, top_class]
    grads = tape.gradient(loss, embeddings)
    importance = tf.norm(grads[0], axis=1).numpy()

    valid_imp = importance[:len(raw_words)]
    if valid_imp.max() > 0:
        valid_imp_norm = valid_imp / valid_imp.max()
    else:
        valid_imp_norm = np.zeros_like(valid_imp)

    cmap = plt.colormaps['Reds']
    html_parts = ['<div style="line-height:2.5; font-size:16px;">']
    for i, word in enumerate(raw_words):
        is_known = (seq[i] > 0)
        if is_known:
            alpha = valid_imp_norm[i]
            r, g, b, _ = cmap(alpha)
            bg_color = f"rgba({int(r*255)},{int(g*255)},{int(b*255)},1.0)"
            title = ""
        else:
            bg_color = "#f0f0f0"
            title = "неизв"
        html_parts.append(
            f'<span style="background-color:{bg_color}; color:black; '
            f'padding:4px 6px; margin:2px; border-radius:4px; '
            f'display:inline-block;" title="{title}">{word}</span>'
        )
    html_parts.append('</div>')
    return "".join(html_parts)

def plot_confidence(proba, class_names, model_name):
    """Гистограмма вероятностей для всех классов."""
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    ax.bar(class_names, proba, color=colors)
    ax.set_ylim([0, 1])
    ax.set_ylabel("Вероятность")
    ax.set_title(f"Уверенность {model_name}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

# ------------------------------------------------------------
# 5. Загрузка примеров новостей из tg_channels.feather
# ------------------------------------------------------------
FEATHER_PATH = Path("Datasets/tg_channels.feather")  # или "tg_channels.feather", если в корне

@st.cache_data
def load_news_from_feather(max_items=500):
    """Загружает датафрейм с колонками 'content' и 'label' из файла tg_channels.feather.
    Возвращает DataFrame или None при ошибке."""
    if not FEATHER_PATH.exists():
        st.warning(f"Файл {FEATHER_PATH} не найден. Используются запасные примеры.")
        return None

    df = pd.read_feather(FEATHER_PATH)
    required_cols = ['content', 'label']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Колонка '{col}' не найдена в tg_channels.feather. Используются запасные примеры.")
            return None

    # Оставляем только нужные столбцы и удаляем пропуски
    df = df[required_cols].dropna(subset=['content']).copy()
    df['content'] = df['content'].astype(str)

    if len(df) > max_items:
        df = df.sample(n=max_items, random_state=42)

    if df.empty:
        st.warning("Датасет пуст. Используются запасные примеры.")
        return None

    return df

# Запасные примеры, если файла нет
FALLBACK_NEWS = [
    {"content": "Биткоин обновил исторический максимум, достигнув отметки в $100000.", "label": 0},
    {"content": "Новая коллекция Gucci весна-лето 2025 поражает яркими красками.", "label": 1},
    {"content": "Сборная России по футболу одержала победу в товарищеском матче.", "label": 2},
    {"content": "Искусственный интеллект помогает диагностировать редкие заболевания.", "label": 3},
    {"content": "Центробанк повысил ключевую ставку до 18% годовых.", "label": 4},
]

def get_label_display(label):
    """Преобразует метку (число или строку) в читаемое имя с эмодзи."""
    CLASS_NAMES = ["Крипта", "Мода", "Спорт", "Технологии", "Финансы"]
    EMOJI_MAP = {0: "💰", 1: "👗", 2: "⚽", 3: "💻", 4: "📈"}
    try:
        label_int = int(label)
        if 0 <= label_int <= 4:
            return f"{CLASS_NAMES[label_int]} {EMOJI_MAP[label_int]}"
    except:
        pass
    return str(label)

# ------------------------------------------------------------
# 6. Интерфейс страницы
# ------------------------------------------------------------
st.set_page_config(page_title="Классификация тематик новостей", page_icon="📰")
st.title("📰 Классификация тематик новостей Telegram")

# Инициализация состояний
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""
if 'random_count' not in st.session_state:
    st.session_state.random_count = 0
if 'true_label' not in st.session_state:
    st.session_state.true_label = None

with st.spinner("Загружаем модели..."):
    vectorizer = load_tfidf_vectorizer()
    logreg = load_logreg()
    rf = load_rf()
    lstm_tok = load_lstm_tokenizer()
    lstm_model = load_lstm_model()
    bert_tok, bert_model = load_bert()

# Загружаем датасет или fallback
SAMPLE_DF = load_news_from_feather()
if SAMPLE_DF is None:
    SAMPLE_DF = pd.DataFrame(FALLBACK_NEWS)

if 'selected_mode' not in st.session_state:
    st.session_state.selected_mode = 'all'
if 'last_text' not in st.session_state:
    st.session_state.last_text = ''
if 'preds' not in st.session_state:
    st.session_state.preds = None

# Кнопка случайной новости
col_rand, _ = st.columns([0.2, 0.8])
with col_rand:
    if st.button("🎲 Случайная новость"):
        # Выбираем случайную строку из SAMPLE_DF
        rand_row = SAMPLE_DF.sample(1).iloc[0]
        st.session_state.input_text = rand_row['content']
        st.session_state.true_label = rand_row['label']
        st.session_state.random_count += 1
        st.session_state.last_text = ''
        st.session_state.preds = None
        st.rerun()

# Отображение реальной метки, если есть
if st.session_state.true_label is not None:
    st.markdown(f"**🏷️ Настоящая тематика:** {get_label_display(st.session_state.true_label)}")
else:
    st.markdown("**🏷️ Настоящая тематика:** не выбрана")

# Поле ввода
text = st.text_area(
    "📝 Введите текст новости:",
    value=st.session_state.input_text,
    height=250,
    key=f"text_input_widget_{st.session_state.random_count}",
    placeholder="Пример: Биткоин обновил исторический максимум..."
)
st.session_state.input_text = text

# Кнопки выбора моделей
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button("Logistic Regression"):
        st.session_state.selected_mode = 'Logistic Regression'
with col2:
    if st.button("Random Forest"):
        st.session_state.selected_mode = 'Random Forest'
with col3:
    if st.button("LSTM"):
        st.session_state.selected_mode = 'LSTM'
with col4:
    if st.button("ImprovedTinyBERT"):
        st.session_state.selected_mode = 'ImprovedTinyBERT'
with col5:
    if st.button("📊 Все модели"):
        st.session_state.selected_mode = 'all'

st.divider()

# Определяем имена классов
CLASS_NAMES = ["Крипта", "Мода", "Спорт", "Технологии", "Финансы"]
EMOJI_MAP = {0: "💰", 1: "👗", 2: "⚽", 3: "💻", 4: "📈"}

# Вычисление предсказаний
if text.strip():
    if text != st.session_state.last_text or st.session_state.preds is None:
        with st.spinner("Выполняется предсказание..."):
            results = {}
            # LogReg
            pred_lr, proba_lr, t_lr = predict_sklearn(logreg, vectorizer, text)
            results['Logistic Regression'] = {
                'pred': pred_lr, 'proba': proba_lr, 'time': t_lr,
                'class_name': CLASS_NAMES[pred_lr], 'emoji': EMOJI_MAP[pred_lr]
            }
            # RF
            pred_rf, proba_rf, t_rf = predict_sklearn(rf, vectorizer, text)
            results['Random Forest'] = {
                'pred': pred_rf, 'proba': proba_rf, 'time': t_rf,
                'class_name': CLASS_NAMES[pred_rf], 'emoji': EMOJI_MAP[pred_rf]
            }
            # LSTM
            if lstm_model is not None:
                pred_lstm, proba_lstm, t_lstm = predict_lstm(lstm_model, lstm_tok, text)
                if pred_lstm is not None:
                    results['LSTM'] = {
                        'pred': pred_lstm, 'proba': proba_lstm, 'time': t_lstm,
                        'class_name': CLASS_NAMES[pred_lstm], 'emoji': EMOJI_MAP[pred_lstm]
                    }
                else:
                    results['LSTM'] = None
            else:
                results['LSTM'] = None
            # BERT
            pred_bert, proba_bert, t_bert, input_ids, attentions = predict_bert(bert_tok, bert_model, text)
            results['ImprovedTinyBERT'] = {
                'pred': pred_bert, 'proba': proba_bert, 'time': t_bert,
                'class_name': CLASS_NAMES[pred_bert], 'emoji': EMOJI_MAP[pred_bert],
                'input_ids': input_ids, 'attentions': attentions
            }
            st.session_state.preds = results
            st.session_state.last_text = text
    else:
        results = st.session_state.preds

    # Отображение в зависимости от выбранного режима
    if st.session_state.selected_mode == 'all':
        st.subheader("📋 Сводка по всем моделям")
        cols = st.columns(4)

        m = results['Logistic Regression']
        cols[0].metric("Logistic Regression", f"{m['class_name']} {m['emoji']}")
        cols[0].write(f"Уверенность: {m['proba'][m['pred']]:.3f}")
        cols[0].write(f"Время: {m['time']:.4f} мс")

        m = results['Random Forest']
        cols[1].metric("Random Forest", f"{m['class_name']} {m['emoji']}")
        cols[1].write(f"Уверенность: {m['proba'][m['pred']]:.3f}")
        cols[1].write(f"Время: {m['time']:.4f} мс")

        if results['LSTM'] is not None:
            m = results['LSTM']
            cols[2].metric("LSTM", f"{m['class_name']} {m['emoji']}")
            cols[2].write(f"Уверенность: {m['proba'][m['pred']]:.3f}")
            cols[2].write(f"Время: {m['time']:.4f} мс")
        else:
            cols[2].warning("LSTM недоступна")

        m = results['ImprovedTinyBERT']
        cols[3].metric("ImprovedTinyBERT", f"{m['class_name']} {m['emoji']}")
        cols[3].write(f"Уверенность: {m['proba'][m['pred']]:.3f}")
        cols[3].write(f"Время: {m['time']:.4f} мс")

        st.subheader("📊 Уверенность моделей в предсказании")
        plot_cols = st.columns(4)
        for i, name in enumerate(['Logistic Regression', 'Random Forest', 'LSTM', 'ImprovedTinyBERT']):
            m = results.get(name)
            if m:
                with plot_cols[i]:
                    plot_confidence(m['proba'], CLASS_NAMES, name)
    else:
        model_name = st.session_state.selected_mode
        model_data = results.get(model_name)
        if model_data is None:
            st.warning(f"Модель {model_name} недоступна.")
        else:
            st.header(f"{model_name}")

            # Визуализация
            if model_name == 'ImprovedTinyBERT':
                st.subheader("🧠 Карта внимания")
                st.markdown(
                    highlight_attention(bert_tok, model_data['input_ids'],
                                        model_data['attentions'], text),
                    unsafe_allow_html=True
                )
                st.caption("Тёплые цвета — высокий вес внимания слова.")
            if model_name == 'LSTM':
                st.subheader("🔍 Важность токенов (градиентная карта)")
                if lstm_model is not None:
                    html = lstm_token_importance(lstm_model, lstm_tok, text)
                    st.markdown(html, unsafe_allow_html=True)
                    st.caption("Яркость – влияние слова. Серые – слова вне словаря.")

            col_left, col_right = st.columns([0.45, 0.55])
            with col_left:
                st.metric("Предсказание", f"{model_data['class_name']} {model_data['emoji']}")
                st.markdown(f"**Уверенность:** {model_data['proba'][model_data['pred']]:.3f}")
                st.markdown(f"**Время инференса:** {model_data['time']:.4f} мс")
            with col_right:
                plot_confidence(model_data['proba'], CLASS_NAMES, model_name)

# Сводная таблица метрик (всегда видна внизу)
st.divider()
st.subheader("📈 Сравнение моделей на тестовой выборке (5 классов)")

metrics = {
    "Модель": ["Logistic Regression", "Random Forest", "LSTM", "ImprovedTinyBERT"],
    "Accuracy": [0.8967, 0.8056, 0.5359, 0.9387],
    "F1‑macro": [0.8988, 0.8140, 0.4724, 0.9404],
    "Inference (ms)": [0.0072, 0.1241, 9.1284, 0.2825],
}
st.dataframe(metrics, use_container_width=True)