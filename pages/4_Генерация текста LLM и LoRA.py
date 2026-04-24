import streamlit as st

st.set_page_config(page_title="Генератор текста", layout="centered")

st.title("Генератор текста")

# Инициализация состояния
if 'video_playing' not in st.session_state:
    st.session_state.video_playing = False

# Кнопка
if st.button(" Сгенерировать текст", type="primary"):
    st.session_state.video_playing = True

# Отображаем видео, если кнопка была нажата
if st.session_state.video_playing:
    st.markdown("Текст для вас можно сгенерировать только на локальной версии Streamlit")
    
    # Прямая ссылка на видео с Hugging Face
    video_url = "https://huggingface.co/prihoslo/qwery_pers/resolve/main/rutube_video_fef8a7569e09458316f2def123beb79c.mp4"
    
    # Воспроизводим видео с автозапуском
    st.video(video_url, format="video/mp4", autoplay=True)
    
    # Добавляем кнопку для сброса
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Сбросить", use_container_width=True):
            st.session_state.video_playing = False
            st.rerun()
    with col2:
        if st.button("🔁 Сгенерировать снова", use_container_width=True):
            st.session_state.video_playing = False
            st.rerun()
    
