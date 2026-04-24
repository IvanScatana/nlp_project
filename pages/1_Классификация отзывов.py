import os
import json

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
import numpy as np
import time
import joblib
import requests
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel

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
# 2. Кастомная модель ImprovedTinyBERTFull
# ------------------------------------------------------------
class ImprovedTinyBERTFull(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3, hidden_dim=512):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            "cointegrated/rubert-tiny2",
            attn_implementation="eager"          # ← ОБЯЗАТЕЛЬНО для output_attentions
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
        # Всегда запрашиваем attention, чтобы визуализация работала надёжно
        bert_out = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask,
                             output_attentions=True)   # <-- принудительно True
        last_hidden = bert_out.last_hidden_state
        attentions = bert_out.attentions               # всегда будет кортеж

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
# 3. Загрузка всех моделей и вспомогательных объектов
# ------------------------------------------------------------
@st.cache_resource
def load_tfidf_vectorizer():
    path = download_file("tfidf_vectorizer.pkl")
    return joblib.load(path)

@st.cache_resource
def load_logreg():
    path = download_file("logreg_tfidf.pkl")
    return joblib.load(path)

@st.cache_resource
def load_rf():
    path = download_file("rf_tfidf.pkl")
    return joblib.load(path)

@st.cache_resource
def load_lstm_tokenizer():
    path = download_file("tokenizer_lstm.pkl")
    return joblib.load(path)

@st.cache_resource
def load_lstm_model():
    path = download_file("best_lstm_model.keras")
    try:
        model = tf.keras.models.load_model(path, compile=False)
        # Инициализируем модель, чтобы стали доступны .inputs и .output
        dummy = tf.keras.preprocessing.sequence.pad_sequences([[0]], maxlen=128)
        _ = model.predict(dummy, verbose=0)
        _ = model(tf.constant(dummy))
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки LSTM: {e}")
        return None

@st.cache_resource
def load_bert():
    # Эти файлы нужны токенизатору, они уже скачаны в CACHE_DIR
    download_file("tokenizer.json")
    download_file("tokenizer_config.json")
    model_pt = download_file("best_ImprovedTinyBERTFull.pt")

    tokenizer = AutoTokenizer.from_pretrained(str(CACHE_DIR))
    model = ImprovedTinyBERTFull(num_classes=2, dropout=0.3, hidden_dim=512)

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
    if model is None or tokenizer is None:
        return None, None, None
    try:
        t0 = time.perf_counter()
        seq = tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq,
                                                                maxlen=max_len,
                                                                padding='post')
        proba = model.predict(padded, verbose=0)[0]
        pred = np.argmax(proba)
        elapsed = (time.perf_counter() - t0) * 1000
        return pred, proba, elapsed
    except Exception as e:
        st.warning(f"Ошибка LSTM: {e}")
        return None, None, None

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

# ------------------------------------------------------------
# 5. Визуализация внимания BERT и градиентная карта LSTM
# ------------------------------------------------------------
def highlight_attention(tokenizer, input_ids, attentions, text, layer=-1, head=0):
    """
    Группирует BPE-токены в целые слова и показывает внимание от CLS‑токена.
    Служебные токены исключены, цвет текста всегда чёрный.
    """
    if attentions is None or len(attentions) == 0:
        return "<div>Карта внимания недоступна.</div>"
    try:
        # Получаем соответствие токен → слово (word_ids)
        encoded = tokenizer(text, return_offsets_mapping=True, truncation=True, padding=True)
        word_ids = encoded.word_ids()

        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        attn = attentions[layer][0, head, 0, :].detach().numpy()

        # Группируем внимание по словам
        word_attn = {}
        word_token_ids = {}
        for i, tok in enumerate(tokens):
            wid = word_ids[i]
            if wid is None:          # спецтокен ([CLS], [SEP], [PAD])
                continue
            if wid not in word_attn:
                word_attn[wid] = []
                word_token_ids[wid] = []
            word_attn[wid].append(attn[i])
            word_token_ids[wid].append(input_ids[i].item())

        # Собираем слова и усреднённые скоры
        words = []
        scores = []
        for wid in sorted(word_attn):
            word_str = tokenizer.decode(word_token_ids[wid], skip_special_tokens=True).strip()
            word_str = word_str.replace(' ', '')      # убираем пробелы между подсловами
            words.append(word_str)
            scores.append(np.mean(word_attn[wid]))

        scores = np.array(scores)
        norm = mcolors.Normalize(vmin=scores.min(), vmax=scores.max())
        cmap = plt.colormaps['Reds']

        html_parts = ['<div style="line-height:2.5; font-size:16px;">']
        for word, score in zip(words, scores):
            alpha = norm(score)
            r, g, b, _ = cmap(alpha)
            color = f"rgba({int(r*255)},{int(g*255)},{int(b*255)},1.0)"
            html_parts.append(
                f'<span style="background-color:{color}; color:black; '
                f'padding:4px 6px; margin:2px; border-radius:4px; '
                f'display:inline-block;">{word}</span>'
            )
        html_parts.append('</div>')
        return "".join(html_parts)
    except Exception as e:
        return f"<div>Ошибка визуализации: {e}</div>"

def lstm_token_importance(model, tokenizer, text, max_len=128):
    """
    Градиентная карта важности слов для LSTM.
    Показывает все слова исходного текста. Неизвестные слова — серый фон.
    """
    if model is None:
        return "<div>Модель LSTM недоступна.</div>"

    # Извлекаем размер словаря Embedding
    emb_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Embedding):
            emb_layer = layer
            break
    if emb_layer is None:
        return "<div>Embedding слой не найден</div>"
    vocab_size = emb_layer.input_dim  # максимально допустимый индекс + 1

    # Разбиваем текст на слова через встроенную Keras‑утилиту
    from tensorflow.keras.preprocessing.text import text_to_word_sequence
    raw_words = text_to_word_sequence(text)
    if not raw_words:
        return "<div>Не удалось выделить слова.</div>"

    # Преобразуем в индексы, обрезая до vocab_size
    seq = []
    for word in raw_words:
        idx = tokenizer.word_index.get(word, 0)
        if idx >= vocab_size:
            idx = 0        # неизвестное слово
        seq.append(idx)

    # Паддинг и модель
    padded = tf.keras.preprocessing.sequence.pad_sequences([seq],
                                                            maxlen=max_len,
                                                            padding='post')
    input_tensor = tf.constant(padded, dtype=tf.int32)
    _ = model(input_tensor)  # принудительная инициализация графа

    # Строим часть модели после эмбеддинга для градиентов
    tail_layers = model.layers[model.layers.index(emb_layer) + 1:]
    tail_model = tf.keras.Sequential(tail_layers)
    dummy_emb = emb_layer(tf.constant([[0]]))
    _ = tail_model(dummy_emb)

    with tf.GradientTape() as tape:
        embeddings = emb_layer(input_tensor)
        tape.watch(embeddings)
        preds = tail_model(embeddings)
        top_class = tf.argmax(preds[0])
        loss = preds[0, top_class]

    grads = tape.gradient(loss, embeddings)
    importance = tf.norm(grads[0], axis=1).numpy()   # форма (max_len,)

    # Берём только реальные слова (до паддинга)
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
            bg_color = "#f0f0f0"   # светло-серый
            title = "неизв"
        html_parts.append(
            f'<span style="background-color:{bg_color}; color:black; '
            f'padding:4px 6px; margin:2px; border-radius:4px; '
            f'display:inline-block;" title="{title}">{word}</span>'
        )
    html_parts.append('</div>')
    return "".join(html_parts)

def plot_confidence(proba, model_name):
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.bar(["Негативный", "Позитивный"], proba, color=["#ff4d4d", "#4dff4d"])
    ax.set_ylim([0, 1])
    ax.set_ylabel("Вероятность")
    ax.set_title(f"Уверенность {model_name}")
    st.pyplot(fig)

# ------------------------------------------------------------
# 6. Загрузка отзывов из локального JSONL-файла
# ------------------------------------------------------------
FALLBACK_REVIEWS = [
    "В регистратуре были очень грубые, хамоватые сотрудницы. Врач ничего не объяснил, я осталась недовольна.",
    "Отличная поликлиника! Приняли быстро, доктор внимательный, всё подробно рассказал. Рекомендую.",
    "В очереди просидел почти час, но сам приём прошёл хорошо, врач профессионал.",
    "Ужасное отношение! Не вернусь сюда больше. Сплошное разочарование.",
    "Персонал приветливый, чисто, оборудование новое. Очень понравилось.",
    "Неплохая поликлиника, но есть куда расти. Регистратура работает медленно.",
    "Замечательный врач и медсестры! Внимательные и заботливые. Огромное спасибо.",
    "Полный бардак, запись потеряли, ждал зря. Никому не советую.",
    "Быстро, чётко, без проволочек – прекрасный сервис.",
    "Обслуживание среднее, но в целом неплохо. Могло быть и хуже."
]

@st.cache_data
def load_reviews_from_jsonl(filename="Datasets/healthcare_facilities_reviews.jsonl"):
    """Загружает отзывы из JSONL-файла, лежащего в корне приложения.
       Каждая строка файла – JSON с обязательным ключом 'content'.
       Если файл не найден или пуст, возвращает примеры по умолчанию."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reviews = []
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    if 'content' in data:
                        reviews.append(data['content'])
            if not reviews:
                st.warning("Файл с отзывами пуст, используются примеры по умолчанию.")
                return FALLBACK_REVIEWS
            return reviews
    except FileNotFoundError:
        st.warning(f"Файл {filename} не найден. Используются примеры по умолчанию.")
        return FALLBACK_REVIEWS
    except Exception as e:
        st.warning(f"Ошибка чтения файла отзывов: {e}. Используются примеры по умолчанию.")
        return FALLBACK_REVIEWS

# ------------------------------------------------------------
# 7. Интерфейс Streamlit
# ------------------------------------------------------------
st.set_page_config(page_title="Классификация отзывов пациентов", page_icon="📊")
st.title("📊 Сентимент‑анализ отзывов пациентов")

# Инициализация состояний
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""
if 'random_count' not in st.session_state:
    st.session_state.random_count = 0

# Загрузка моделей и данных
with st.spinner("Загружаем модели..."):
    vectorizer = load_tfidf_vectorizer()
    logreg = load_logreg()
    rf = load_rf()
    lstm_tok = load_lstm_tokenizer()
    lstm_model = load_lstm_model()
    bert_tok, bert_model = load_bert()

SAMPLE_REVIEWS = load_reviews_from_jsonl()

if 'selected_mode' not in st.session_state:
    st.session_state.selected_mode = 'all'
if 'last_text' not in st.session_state:
    st.session_state.last_text = ''
if 'preds' not in st.session_state:
    st.session_state.preds = None

# ==================== КНОПКА СЛУЧАЙНОГО ОТЗЫВА ====================
col_rand, _ = st.columns([0.2, 0.8])
with col_rand:
    if st.button("🎲 Случайный отзыв"):
        st.session_state.input_text = np.random.choice(SAMPLE_REVIEWS)
        st.session_state.random_count += 1
        st.session_state.last_text = ''
        st.session_state.preds = None
        st.rerun()

# ==================== ПОЛЕ ВВОДА ====================
text = st.text_area(
    "📝 Введите текст отзыва:",
    value=st.session_state.input_text,
    height=250,
    key=f"text_input_widget_{st.session_state.random_count}",
    placeholder="Пример: В регистратуре нагрубили, но врач замечательный..."
)
st.session_state.input_text = text

# ==================== КНОПКИ ВЫБОРА РЕЖИМА ====================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("Logistic Regression", key="btn_lr"):
        st.session_state.selected_mode = 'Logistic Regression'
with col2:
    if st.button("Random Forest", key="btn_rf"):
        st.session_state.selected_mode = 'Random Forest'
with col3:
    if st.button("LSTM", key="btn_lstm"):
        st.session_state.selected_mode = 'LSTM'
with col4:
    if st.button("ImprovedTinyBERT", key="btn_bert"):
        st.session_state.selected_mode = 'ImprovedTinyBERT'
with col5:
    if st.button("📊 Все модели", key="btn_all"):
        st.session_state.selected_mode = 'all'

st.divider()

# ==================== ВЫЧИСЛЕНИЕ ПРЕДСКАЗАНИЙ ====================
if text.strip():
    if text != st.session_state.last_text or st.session_state.preds is None:
        with st.spinner("Выполняется предсказание..."):
            emoji_map = {0: "😡", 1: "😊"}
            class_names = {0: "Негативный", 1: "Позитивный"}

            results = {}
            # LogReg
            pred_lr, proba_lr, t_lr = predict_sklearn(logreg, vectorizer, text)
            results['Logistic Regression'] = {
                'pred': pred_lr,
                'proba': proba_lr,
                'time': t_lr,
                'class_name': class_names[pred_lr],
                'emoji': emoji_map[pred_lr]
            }
            # RF
            pred_rf, proba_rf, t_rf = predict_sklearn(rf, vectorizer, text)
            results['Random Forest'] = {
                'pred': pred_rf,
                'proba': proba_rf,
                'time': t_rf,
                'class_name': class_names[pred_rf],
                'emoji': emoji_map[pred_rf]
            }
            # LSTM
            if lstm_model is not None:
                pred_lstm, proba_lstm, t_lstm = predict_lstm(lstm_model, lstm_tok, text)
                if pred_lstm is not None:
                    results['LSTM'] = {
                        'pred': pred_lstm,
                        'proba': proba_lstm,
                        'time': t_lstm,
                        'class_name': class_names[pred_lstm],
                        'emoji': emoji_map[pred_lstm]
                    }
                else:
                    results['LSTM'] = None
            else:
                results['LSTM'] = None
            # BERT
            pred_bert, proba_bert, t_bert, input_ids, attentions = predict_bert(bert_tok, bert_model, text)
            results['ImprovedTinyBERT'] = {
                'pred': pred_bert,
                'proba': proba_bert,
                'time': t_bert,
                'class_name': class_names[pred_bert],
                'emoji': emoji_map[pred_bert],
                'input_ids': input_ids,
                'attentions': attentions
            }

            st.session_state.preds = results
            st.session_state.last_text = text
    else:
        results = st.session_state.preds
        emoji_map = {0: "😡", 1: "😊"}
        class_names = {0: "Негативный", 1: "Позитивный"}
        for m_data in results.values():
            if m_data:
                m_data['class_name'] = class_names[m_data['pred']]
                m_data['emoji'] = emoji_map[m_data['pred']]

    # ==================== ОТОБРАЖЕНИЕ В ЗАВИСИМОСТИ ОТ РЕЖИМА ====================
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
                    plot_confidence(m['proba'], name)

    else:
        model_name = st.session_state.selected_mode
        model_data = results.get(model_name)

        if model_data is None:
            st.warning(f"Модель {model_name} недоступна.")
        else:
            st.header(f"{model_name}")

            # Визуализация внимания (перед метриками)
            if model_name == 'ImprovedTinyBERT':
                st.subheader("🧠 Карта внимания (агрегировано по словам)")
                st.markdown(
                    highlight_attention(bert_tok, model_data['input_ids'],
                                        model_data['attentions'], text),
                    unsafe_allow_html=True,
                )
                st.caption("Тёплые цвета — высокий вес внимания слова.")

            if model_name == 'LSTM':
                st.subheader("🔍 Важность токенов (градиентная карта)")
                if lstm_model is not None:
                    html = lstm_token_importance(lstm_model, lstm_tok, text)
                    st.markdown(html, unsafe_allow_html=True)
                    st.caption("Яркость – влияние слова. Серые – слова вне словаря модели (неизв).")
                else:
                    st.warning("Модель LSTM недоступна для визуализации.")

            # Метрики и график уверенности
            col_left, col_right = st.columns([0.45, 0.55])
            with col_left:
                st.metric("Предсказание", f"{model_data['class_name']} {model_data['emoji']}")
                st.markdown(f"**Уверенность:** {model_data['proba'][model_data['pred']]:.3f}")
                st.markdown(f"**Время инференса:** {model_data['time']:.4f} мс")
            with col_right:
                plot_confidence(model_data['proba'], model_name)

# ==================== СВОДНАЯ ТАБЛИЦА МЕТРИК ====================
st.divider()
st.subheader("📈 Сравнение моделей на тестовой выборке")

metrics = {
    "Модель": [
        "Logistic Regression",
        "Random Forest",
        "LSTM",
        "ImprovedTinyBERT",
    ],
    "Accuracy": [0.9503, 0.8885, 0.9307, 0.9101],
    "F1‑macro": [0.9489, 0.8868, 0.9288, 0.9073],
    "Inference (ms)": [0.0003, 0.0298, 2.1639, 0.1699],
}

st.dataframe(metrics, width='stretch')