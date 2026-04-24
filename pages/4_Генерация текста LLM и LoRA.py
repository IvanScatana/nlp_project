import streamlit as st
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from PIL import Image
import requests
from io import BytesIO
from huggingface_hub import snapshot_download
import base64

# Конфигурация страницы
st.set_page_config(
    page_title="Qwen2.5-7B с персонажами",
    page_icon="🤖",
    layout="wide"
)

# Константы
BASE_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_REPO = "prihoslo/qwery_pers"

# Конфигурация персонажей с обновленными ссылками
CHARACTERS = {
    "Йода": {
        "adapter_subfolder": "qwen-yoda-lora",
        "system_prompt": "Ты мастер Йода. отвечай как мудрый старый джедай.",
        "image_url": "https://huggingface.co/prihoslo/qwery_pers/resolve/main/ce315119a36e31077daf1bea09d18dad.jpg",
        "image_type": "static",
        "description": "Мудрый джедай-мастер"
    },
    "Горлум": {
        "adapter_subfolder": "qwen-Gorlum-lora",
        "system_prompt": "Ты Горлум из фильма властелин колец. отвечай на вопросы и комментируй фразы как он.",
        "image_url": "https://huggingface.co/prihoslo/qwery_pers/resolve/main/download.jpeg",
        "image_type": "static",
        "description": "Существо из Властелина Колец"
    },
    "АУФ!": {
        "adapter_subfolder": "qwen-bro-lora",
        "system_prompt": "Ты — настоящий пацан, брат и волк. Отвечай на вопросы глубокомысленными пацанскими цитатами.",
        "image_url": "https://huggingface.co/prihoslo/qwery_pers/resolve/main/wolf-dance.gif",
        "image_type": "gif",
        "description": "Настоящий пацан с района"
    },
    "Без адаптера": {
        "adapter_subfolder": None,
        "system_prompt": "",
        "image_url": "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png",
        "image_type": "static",
        "description": "Базовая модель Qwen2.5-7B"
    }
}

def cleanup_memory():
    """Очистка памяти GPU и CPU"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

@st.cache_resource
def load_base_model():
    """Загрузка только базовой модели (выполняется один раз)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with st.spinner("🚀 Загрузка базовой модели Qwen2.5-7B..."):
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        base_model.eval()
        
    return base_model, tokenizer, device

def load_adapter_to_model(base_model, adapter_subfolder):
    """Загрузка адаптера в существующую модель"""
    try:
        # Загружаем адаптер
        model = PeftModel.from_pretrained(
            base_model,
            ADAPTER_REPO,
            subfolder=adapter_subfolder
        )
        model.eval()
        return model
    except Exception as e:
        st.warning(f"Не удалось загрузить адаптер из репозитория: {e}")
        
        # Альтернативный вариант - скачать локально
        try:
            with st.spinner(f"Скачиваем адаптер {adapter_subfolder} локально..."):
                local_path = snapshot_download(
                    repo_id=ADAPTER_REPO,
                    allow_patterns=f"{adapter_subfolder}/*",
                    local_dir=f"./temp_adapters/{adapter_subfolder}"
                )
                adapter_local_path = f"./temp_adapters/{adapter_subfolder}/{adapter_subfolder}"
                model = PeftModel.from_pretrained(base_model, adapter_local_path)
                model.eval()
                return model
        except Exception as e2:
            st.error(f"Не удалось загрузить адаптер: {e2}")
            return base_model

def load_character_image(url, image_type="static"):
    """Загрузка изображения персонажа с поддержкой GIF"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            if image_type == "gif":
                # Для GIF возвращаем байты для использования в markdown
                return response.content
            else:
                # Для статических изображений используем PIL
                return Image.open(BytesIO(response.content))
    except Exception as e:
        st.warning(f"Не удалось загрузить изображение: {e}")
    return None

def display_character_image(image_data, image_type, character_name):
    """Отображение изображения персонажа с заданными размерами"""
    if image_data:
        if image_type == "gif":
            # Для GIF используем HTML для отображения анимации
            gif_base64 = base64.b64encode(image_data).decode("utf-8")
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center; margin: 20px 0;">
                    <div style="text-align: center;">
                        <img src="data:image/gif;base64,{gif_base64}" 
                             alt="{character_name}" 
                             style="max-width: 300px; 
                                    max-height: 300px; 
                                    width: auto; 
                                    height: auto; 
                                    border-radius: 15px; 
                                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                                    object-fit: contain;">
                        <p style="margin-top: 10px; font-size: 18px; font-weight: bold; color: #ff4b4b;">
                            {character_name}
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Для статических изображений используем st.image
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image_data, caption=character_name, use_container_width=True)
    else:
        st.warning("Не удалось загрузить изображение")

def generate_response(model, tokenizer, messages, temperature, top_p, max_tokens):
    """Генерация ответа модели"""
    # Применение шаблона чата Qwen
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Параметры генерации
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Извлечение ответа
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# Инициализация session state
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'current_char' not in st.session_state:
    st.session_state.current_char = None
if 'base_model_loaded' not in st.session_state:
    st.session_state.base_model_loaded = False

# Интерфейс Streamlit
st.title("🎭 Qwen2.5-7B с персонажами")
st.markdown("Выберите персонажа для общения и настройте параметры генерации")

# Боковая панель с настройками
with st.sidebar:
    st.header("⚙️ Параметры генерации")
    
    temperature = st.slider(
        "Temperature (креативность)",
        min_value=0.1,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Чем выше значение, тем более креативные и случайные ответы"
    )
    
    top_p = st.slider(
        "Top-p (ядерная выборка)",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Меньшие значения делают ответы более фокусированными"
    )
    
    max_tokens = st.slider(
        "Max New Tokens",
        min_value=50,
        max_value=2048,
        value=512,
        step=50,
        help="Максимальная длина ответа"
    )
    
    st.markdown("---")
    
    # Выбор персонажа
    st.header("🎪 Выбор персонажа")
    selected_char = st.radio(
        "Выберите персонажа:",
        list(CHARACTERS.keys()),
        format_func=lambda x: f"{x} - {CHARACTERS[x]['description']}"
    )
    
    # Информация о модели
    st.markdown("---")
    st.markdown("**Информация:**")
    char_config = CHARACTERS[selected_char]
    if char_config["adapter_subfolder"]:
        st.text(f"Адаптер: {char_config['adapter_subfolder']}")
    else:
        st.text("Базовая модель без адаптера")
    
    st.text(f"Temperature: {temperature}")
    st.text(f"Top-p: {top_p}")
    st.text(f"Max tokens: {max_tokens}")
    
    # Добавляем кнопку для принудительной очистки кэша
    st.markdown("---")
    if st.button("🗑️ Очистить кэш моделей", use_container_width=True):
        st.session_state.current_char = None
        st.session_state.current_model = None
        cleanup_memory()
        st.success("Кэш очищен!")
        st.rerun()

# Основная область
col1, col2 = st.columns([3, 1])

with col1:
    # Отображение изображения персонажа на главном экране
    char_config = CHARACTERS[selected_char]
    image_data = load_character_image(char_config["image_url"], char_config.get("image_type", "static"))
    
    # Создаем контейнер для изображения
    image_container = st.container()
    with image_container:
        display_character_image(image_data, char_config.get("image_type", "static"), selected_char)
    
    st.markdown("---")
    
    st.subheader(f"💬 Чат с персонажем: {selected_char}")
    
    # Поле ввода вопроса
    user_question = st.text_area(
        "Ваш вопрос:",
        placeholder=f"Задайте вопрос персонажу {selected_char}...",
        height=100
    )
    
    # Кнопка генерации
    if st.button("🚀 Получить ответ", type="primary", use_container_width=True):
        if not user_question.strip():
            st.warning("Пожалуйста, введите вопрос!")
        else:
            with st.spinner("Генерирую ответ..."):
                try:
                    # Загружаем базовую модель только один раз
                    if not st.session_state.base_model_loaded:
                        base_model, tokenizer, device = load_base_model()
                        st.session_state.base_model = base_model
                        st.session_state.tokenizer = tokenizer
                        st.session_state.device = device
                        st.session_state.base_model_loaded = True
                    else:
                        base_model = st.session_state.base_model
                        tokenizer = st.session_state.tokenizer
                        device = st.session_state.device
                    
                    # Проверяем, нужно ли переключать персонажа
                    if st.session_state.current_char != selected_char:
                        with st.spinner(f"🔄 Загружаем персонажа {selected_char}..."):
                            if char_config["adapter_subfolder"]:
                                # Загружаем новый адаптер
                                st.session_state.current_model = load_adapter_to_model(
                                    base_model, 
                                    char_config["adapter_subfolder"]
                                )
                                st.session_state.current_char = selected_char
                                # Очищаем память от старого адаптера
                                cleanup_memory()
                                st.success(f"✅ Персонаж {selected_char} загружен!")
                            else:
                                # Используем базовую модель без адаптера
                                st.session_state.current_model = base_model
                                st.session_state.current_char = selected_char
                                st.info("📝 Используется базовая модель без адаптера")
                    
                    # Используем текущую модель
                    model = st.session_state.current_model
                    
                    # Формирование сообщений
                    messages = []
                    
                    # Добавление системного промта если есть
                    if char_config["system_prompt"]:
                        messages.append({
                            "role": "system",
                            "content": char_config["system_prompt"]
                        })
                    
                    # Добавление вопроса пользователя
                    messages.append({
                        "role": "user",
                        "content": user_question
                    })
                    
                    # Генерация ответа
                    response = generate_response(
                        model, tokenizer,
                        messages, temperature, top_p, max_tokens
                    )
                    
                    # Отображение ответа
                    st.markdown("### Ответ:")
                    st.markdown(f"> {response}")
                    
                    # Кнопка копирования
                    st.code(response, language="text")
                    
                except Exception as e:
                    st.error(f"Произошла ошибка: {str(e)}")
                    st.exception(e)
                    # Сбрасываем состояние при ошибке
                    st.session_state.current_char = None
                    st.session_state.current_model = None

with col2:
    st.subheader("📋 Системный промт")
    if char_config["system_prompt"]:
        st.info(char_config["system_prompt"])
    else:
        st.info("Используется стандартный промт базовой модели")
    
    st.markdown("---")
    st.subheader("💡 Примеры вопросов")
    examples = [
        "Расскажи о силе",
        "Что ты думаешь о дружбе?",
        "Дай совет как жить",
        "Расскажи историю",
        "Как стать сильнее?",
        "Что такое настоящая дружба?"
    ]
    
    for example in examples:
        if st.button(example, key=f"ex_{example}", use_container_width=True):
            st.session_state['example_question'] = example
    
    if 'example_question' in st.session_state:
        st.text_input("Пример вопроса", value=st.session_state['example_question'])

# Футер
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Модель: Qwen2.5-7B-Instruct | Адаптеры: LoRA | Интерфейс: Streamlit</p>
        <p>💡 При переключении персонажа адаптер загружается динамически</p>
    </div>
    """,
    unsafe_allow_html=True
)