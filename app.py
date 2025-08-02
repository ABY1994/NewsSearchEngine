import streamlit as st
import sqlite3
import os
from PIL import Image
import pytesseract
import numpy as np
import re
from datetime import datetime
import shutil
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import base64
import hashlib
import hmac
import sys
import traceback

# Налаштування шляхів
UPLOAD_DIR = "uploads"
DB_DIR = "dbs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Налаштування Tesseract OCR
try:
    # Для Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except:
    try:
        # Для Linux/Streamlit Cloud
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    except:
        st.warning("Tesseract OCR не знайдено. Розпізнавання тексту на зображеннях недоступне.")

# Глобальний обробник помилок
def handle_exception(exc_type, exc_value, exc_traceback):
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    st.error(f"**Критична помилка:**\n```\n{error_msg}\n```")
    st.info("Додаток буде автоматично перезапущено через 30 секунд...")
    time.sleep(30)
    st.experimental_rerun()

sys.excepthook = handle_exception

# Функція для хешування паролів
def hash_password(password):
    salt = "secure_salt_456"  # Унікальна сіль
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()

# Функція перевірки пароля
def check_password(hashed_password, user_password):
    return hmac.compare_digest(hashed_password, hash_password(user_password))

# Ініціалізація моделі для семантичного пошуку
try:
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
except Exception as e:
    st.error(f"Помилка завантаження моделі ML: {str(e)}")
    model = None

# Ініціалізація баз даних
def init_db():
    for db_name in ['news', 'instructions']:
        conn = sqlite3.connect(os.path.join(DB_DIR, f'{db_name}.db'))
        c = conn.cursor()
        
        c.execute(f'''CREATE TABLE IF NOT EXISTS {db_name}
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT,
                    screenshot_path TEXT,
                    original_link TEXT,
                    additional_links TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        
        c.execute(f'''CREATE TABLE IF NOT EXISTS deleted_{db_name}
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT,
                    screenshot_path TEXT,
                    original_link TEXT,
                    additional_links TEXT,
                    timestamp DATETIME,
                    delete_date DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        
        conn.commit()
        conn.close()

# Функція нормалізації тексту
def normalize_text(text):
    if not text:
        return ""
    text = re.sub(r'[^a-zA-Zа-яА-ЯїЇєЄіІґҐ0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

# Функція додавання до бази
def add_to_db(db_name, description, screenshot, original_link, additional_links=None):
    try:
        conn = sqlite3.connect(os.path.join(DB_DIR, f'{db_name}.db'))
        c = conn.cursor()
        
        screenshot_path = ""
        if screenshot:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            screenshot_path = os.path.join(UPLOAD_DIR, f"{db_name}_{timestamp}.png")
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            with open(screenshot_path, "wb") as f:
                f.write(screenshot.getbuffer())
        
        c.execute(f"INSERT INTO {db_name} (description, screenshot_path, original_link, additional_links) VALUES (?, ?, ?, ?)",
                (description, screenshot_path, original_link, additional_links))
        
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Помилка збереження в базу: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

# Функція пошуку в базі
def search_in_db(query, db_name, num_results=5):
    try:
        conn = sqlite3.connect(os.path.join(DB_DIR, f'{db_name}.db'))
        c = conn.cursor()
        
        c.execute(f"SELECT * FROM {db_name}")
        records = c.fetchall()
        
        if not records or not model:
            return []
        
        texts = []
        for record in records:
            text = record[1]
            if record[2]:
                try:
                    img_text = pytesseract.image_to_string(Image.open(record[2]), lang='ukr+rus')
                    text += " " + normalize_text(img_text)
                except:
                    pass
            texts.append(text)
        
        query_embedding = model.encode([normalize_text(query)])[0]
        doc_embeddings = model.encode(texts)
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]
        return [(records[i], similarities[i]) for i in sorted_indices[:num_results]]
    except Exception as e:
        st.error(f"Помилка пошуку: {str(e)}")
        return []
    finally:
        if conn:
            conn.close()

# Функція видалення запису
def delete_record(record_id, db_name):
    try:
        conn = sqlite3.connect(os.path.join(DB_DIR, f'{db_name}.db'))
        c = conn.cursor()
        c.execute(f"INSERT INTO deleted_{db_name} SELECT *, CURRENT_TIMESTAMP FROM {db_name} WHERE id = ?", (record_id,))
        c.execute(f"DELETE FROM {db_name} WHERE id = ?", (record_id,))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Помилка видалення: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

# Функція відновлення запису
def restore_record(record_id, db_name):
    try:
        conn = sqlite3.connect(os.path.join(DB_DIR, f'{db_name}.db'))
        c = conn.cursor()
        c.execute(f"INSERT INTO {db_name} SELECT id, description, screenshot_path, original_link, additional_links, timestamp FROM deleted_{db_name} WHERE id = ?", (record_id,))
        c.execute(f"DELETE FROM deleted_{db_name} WHERE id = ?", (record_id,))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Помилка відновлення: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

# Головний додаток
def main():
    st.set_page_config(layout="wide", page_title="Інтелектуальний пошук новин та інструкцій")
    
    # Стилі CSS
    st.markdown("""
    <style>
    :root {
        --primary-color: #D8BFD8;
        --secondary-color: #f0f0f0;
        --button-red: #ff4444;
        --button-brown: #D2B48C;
        --button-purple: #4B0082;
        --button-green: #00C853;
        --button-black: #000000;
        --text-white: #FFFFFF;
        --text-black: #000000;
    }
    
    body {
        background-color: var(--secondary-color);
        font-family: Arial, sans-serif;
    }
    
    .header {
        background-color: var(--primary-color);
        padding: 15px 0;
        text-align: center;
        width: 100%;
        margin-bottom: 20px;
    }
    
    .header h1 {
        color: var(--text-white);
        font-weight: bold;
        text-transform: uppercase;
        text-shadow: 1px 1px 2px #00BFFF;
        margin: 0;
        font-size: 2.5rem;
    }
    
    .button {
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.2s;
        border: none;
        padding: 10px 20px;
        cursor: pointer;
        margin: 5px;
        text-align: center;
    }
    
    .button:active {
        transform: scale(0.98);
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) inset;
    }
    
    .search-button {
        background-color: var(--button-red);
        color: var(--text-white);
    }
    
    .add-button {
        background-color: var(--button-brown);
        color: var(--text-white);
        text-shadow: 1px 1px 1px var(--text-black);
    }
    
    .full-db-button {
        background-color: var(--button-purple);
        color: var(--text-white);
    }
    
    .save-button {
        background-color: var(--button-green);
        color: var(--text-white);
    }
    
    .delete-button {
        background-color: var(--button-black);
        color: var(--text-white);
    }
    
    .result-box {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        position: relative;
    }
    
    .similarity-badge {
        position: absolute;
        top: 10px;
        right: 10px;
        border: 2px solid black;
        border-radius: 5px;
        padding: 5px 10px;
        font-weight: bold;
    }
    
    .highlight {
        background-color: yellow;
        padding: 2px;
    }
    
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
        margin-left: 5px;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 250px;
        background-color: #E0E0E0;
        color: #000;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.9rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    .required::before {
        content: "●";
        color: red;
        margin-right: 5px;
    }
    
    .form-section {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .form-title {
        background-color: #90EE90;
        color: white;
        text-shadow: 1px 1px 1px black;
        padding: 8px 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        display: inline-block;
    }
    
    .screenshot-container {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        margin: 20px 0;
    }
    
    .screenshot-item {
        cursor: pointer;
        border: 2px solid #ddd;
        border-radius: 8px;
        overflow: hidden;
        transition: transform 0.3s;
        max-width: 200px;
    }
    
    .screenshot-item:hover {
        transform: scale(1.05);
        border-color: #4285f4;
    }
    
    .screenshot-item img {
        width: 100%;
        height: auto;
        display: block;
    }
    
    .modal {
        display: none;
        position: fixed;
        z-index: 1000;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0,0,0,0.9);
        overflow: auto;
        justify-content: center;
        align-items: center;
    }
    
    .modal-content {
        margin: auto;
        display: block;
        max-width: 90%;
        max-height: 90%;
    }
    
    .close-modal {
        position: absolute;
        top: 20px;
        right: 35px;
        color: #fff;
        font-size: 40px;
        font-weight: bold;
        cursor: pointer;
        transition: 0.3s;
    }
    
    .close-modal:hover {
        color: #f00;
    }
    
    .save-image-btn {
        position: absolute;
        bottom: 20px;
        right: 20px;
        background-color: #4285f4;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-weight: bold;
    }
    
    @media (max-width: 768px) {
        .button-container {
            flex-direction: column;
        }
        
        .button {
            width: 100%;
            margin-bottom: 10px;
        }
    }
    
    .backup-section {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
        border: 1px solid #dee2e6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # JavaScript для модальних вікон
    st.markdown("""
    <script>
    function openModal(imgPath) {
        var modal = document.getElementById('imageModal');
        var modalImg = document.getElementById("modalImage");
        modal.style.display = "flex";
        modalImg.src = imgPath;
    }
    
    function closeModal() {
        document.getElementById('imageModal').style.display = "none";
    }
    
    function saveImage() {
        var img = document.getElementById("modalImage");
        var link = document.createElement('a');
        link.href = img.src;
        link.download = 'screenshot_' + new Date().getTime() + '.png';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
    
    window.onclick = function(event) {
        var modal = document.getElementById('imageModal');
        if (event.target == modal) {
            closeModal();
        }
    }
    </script>
    """, unsafe_allow_html=True)
    
    # Модальне вікно для зображень
    st.markdown("""
    <div id="imageModal" class="modal">
        <span class="close-modal" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImage">
        <div class="save-image-btn" onclick="saveImage()">Зберегти зображення</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Аутентифікація
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.is_admin = False
    
    if not st.session_state.authenticated:
        st.title("🔐 Вхід до системи")
        username = st.text_input("Логін")
        password = st.text_input("Пароль", type="password")
        
        if st.button("Увійти"):
            try:
                if username == st.secrets["auth"]["username"] and check_password(st.secrets["auth"]["password"], password):
                    st.session_state.authenticated = True
                    st.session_state.is_admin = True
                    st.experimental_rerun()
                else:
                    st.error("Невірний логін або пароль")
            except Exception as e:
                st.error(f"Помилка аутентифікації: {str(e)}")
        return
    
    # Ініціалізація баз даних
    try:
        init_db()
    except Exception as e:
        st.error(f"Помилка ініціалізації баз даних: {str(e)}")
        st.info("Спробуємо ще раз через 10 секунд...")
        time.sleep(10)
        st.experimental_rerun()
    
    # Головний інтерфейс
    st.markdown('<div class="header"><h1>ВЕБ-ІНТЕРФЕЙС</h1></div>', unsafe_allow_html=True)
    
    # Кнопки головного меню
    col1, col2, col3 = st.columns(3)
    with col1:
        add_news_btn = st.button("📰 Додати новину", key="add_news_btn", use_container_width=True)
    with col2:
        add_instr_btn = st.button("📝 Додати інструкцію", key="add_instr_btn", use_container_width=True)
    with col3:
        show_all_btn = st.button("🗂️ Вся база новин та інструкцій", key="show_all_btn", use_container_width=True)
    
    # Пошукова панель
    st.markdown("---")
    search_query = st.text_input("🔍 Введіть запит для пошуку:", key="search_query", placeholder="Пошук новин та інструкцій...")
    
    col_search, col_num = st.columns([3, 1])
    with col_search:
        search_btn = st.button("🚀 ПОШУК", key="search_btn", use_container_width=True)
    with col_num:
        num_results = st.selectbox("Кількість результатів:", [5, 7, 10, 12, 15, 20], index=0, key="num_results")
    
    # Обробка пошуку
    if search_btn and search_query:
        if not model:
            st.warning("Модель ML не завантажена. Пошук може працювати некоректно.")
        
        st.session_state.search_type = st.radio("Пошук в:", ["Новини", "Інструкції"], horizontal=True, key="search_type")
        
        db_name = "news" if st.session_state.search_type == "Новини" else "instructions"
        results = search_in_db(search_query, db_name, num_results)
        
        if results:
            st.subheader("Основні результати")
            for (record, score) in results:
                display_record(record, score, db_name, show_delete=st.session_state.is_admin)
            
            # Пошук в іншій базі
            other_db = "instructions" if db_name == "news" else "news"
            other_results = search_in_db(search_query, other_db, 3)
            
            if other_results:
                st.subheader("Інші результати")
                for (record, score) in other_results:
                    display_record(record, score, other_db, show_delete=st.session_state.is_admin)
        else:
            st.warning("Нічого не знайдено. Спробуйте інший запит.")
    
    # Форма додавання новини/інструкції
    if add_news_btn or add_instr_btn:
        db_type = "news" if add_news_btn else "instructions"
        item_type = "новину" if add_news_btn else "інструкцію"
        
        with st.form(f"{db_type}_form", clear_on_submit=True):
            st.markdown(f'<div class="form-title">Додати {item_type}</div>', unsafe_allow_html=True)
            
            # Опис
            st.markdown('<span class="required">Опис</span>', unsafe_allow_html=True)
            description = st.text_area(f"Опис {item_type}:", max_chars=1000, key=f"desc_{db_type}", height=150)
            
            # Скріншот
            st.markdown('<span class="required">Скріншот</span>', unsafe_allow_html=True)
            screenshot = st.file_uploader("Завантажити зображення:", type=["jpg", "png", "jpeg", "gif"], key=f"screen_{db_type}")
            
            # Посилання на оригінал
            st.markdown('<span class="required">Посилання на оригінал</span>', unsafe_allow_html=True)
            original_link = st.text_input("URL:", key=f"orig_link_{db_type}")
            
            # Додаткові посилання
            st.markdown("Додаткові посилання (необов'язково)")
            additional_links = st.text_input("URL:", key=f"add_links_{db_type}")
            
            # Кнопка збереження
            submit = st.form_submit_button(f"💾 Зберегти {item_type}")
            
            if submit:
                if not description or not screenshot or not original_link:
                    st.error("Будь ласка, заповніть всі обов'язкові поля!")
                else:
                    if add_to_db(db_type, description, screenshot, original_link, additional_links):
                        st.success(f"{item_type.capitalize()} успішно додано!")
                        time.sleep(2)
                        st.experimental_rerun()
    
    # Перегляд всієї бази
    if show_all_btn:
        st.subheader("Вся база даних")
        db_choice = st.radio("Переглянути:", ["Новини", "Інструкції", "Видалені матеріали"], horizontal=True, key="db_choice")
        
        if db_choice == "Видалені матеріали":
            st.warning("Цей розділ містить видалені матеріали. Ви можете відновити їх при необхідності.")
            db_type = st.radio("Тип матеріалів:", ["Новини", "Інструкції"], horizontal=True)
            db_name = f"deleted_{'news' if db_type == 'Новини' else 'instructions'}"
        else:
            db_name = "news" if db_choice == "Новини" else "instructions"
        
        try:
            conn = sqlite3.connect(os.path.join(DB_DIR, f"{db_name.split('_')[-1]}.db"))
            c = conn.cursor()
            
            if db_choice == "Видалені матеріали":
                c.execute(f"SELECT * FROM {db_name}")
            else:
                c.execute(f"SELECT * FROM {db_name} ORDER BY timestamp DESC")
            
            records = c.fetchall()
            if records:
                for record in records:
                    display_record(record, None, db_name, 
                                  show_delete=(db_choice != "Видалені матеріали" and st.session_state.is_admin),
                                  show_restore=(db_choice == "Видалені матеріали" and st.session_state.is_admin))
            else:
                st.info("База даних порожня")
        except Exception as e:
            st.error(f"Помилка доступу до бази: {str(e)}")
        finally:
            if conn:
                conn.close()
    
    # Резервне копіювання в бічній панелі
    with st.sidebar:
        st.subheader("🔒 Адміністрування")
        st.markdown(f"**Користувач:** {st.secrets['auth']['username']}")
        
        st.markdown("---")
        st.subheader("🔄 Резервне копіювання")
        
        with st.expander("📥 Завантажити бази даних"):
            try:
                with open(os.path.join(DB_DIR, "news.db"), "rb") as f_news:
                    st.download_button(
                        label="Завантажити базу новин",
                        data=f_news,
                        file_name="news_backup.db",
                        mime="application/octet-stream"
                    )
                
                with open(os.path.join(DB_DIR, "instructions.db"), "rb") as f_instr:
                    st.download_button(
                        label="Завантажити базу інструкцій",
                        data=f_instr,
                        file_name="instructions_backup.db",
                        mime="application/octet-stream"
                    )
            except Exception as e:
                st.error(f"Помилка резервного копіювання: {str(e)}")
        
        st.markdown("---")
        if st.button("🚪 Вийти з системи"):
            st.session_state.authenticated = False
            st.experimental_rerun()

# Функція відображення запису
def display_record(record, score, db_name, show_delete=False, show_restore=False):
    try:
        id, desc, screenshot_path, orig_link, add_links, timestamp = record[:6]
        
        with st.container():
            st.markdown(f"<div class='result-box'>", unsafe_allow_html=True)
            
            # Бейдж зі схожістю
            if score is not None:
                if score >= 0.7:
                    color = "green"
                elif score >= 0.5:
                    color = "goldenrod"
                elif score >= 0.3:
                    color = "orange"
                else:
                    color = "red"
                
                st.markdown(f"<div class='similarity-badge' style='color: {color}'>{score:.2f}</div>", 
                          unsafe_allow_html=True)
            
            # Опис
            st.markdown(f"**Опис:** {desc}")
            
            # Скріншоти
            if screenshot_path and os.path.exists(screenshot_path):
                st.markdown("**Скріншот:**")
                
                # Конвертація зображення
                with open(screenshot_path, "rb") as f:
                    img_data = f.read()
                    img_base64 = base64.b64encode(img_data).decode()
                
                st.markdown(
                    f"<div class='screenshot-container'>"
                    f"<div class='screenshot-item' onclick='openModal(\"data:image/png;base64,{img_base64}\")'>"
                    f"<img src='data:image/png;base64,{img_base64}' alt='Скріншот'>"
                    f"</div></div>",
                    unsafe_allow_html=True
                )
            
            # Посилання
            if orig_link:
                st.markdown(f"**Посилання на оригінал:** [Відкрити]({orig_link})")
            if add_links:
                st.markdown(f"**Додаткові посилання:** [Відкрити]({add_links})")
            
            # Кнопки дій
            col1, col2 = st.columns([1, 3])
            with col1:
                if show_delete:
                    if st.button(f"🗑️ Видалити", key=f"del_{id}_{db_name}"):
                        if delete_record(id, db_name.split('_')[-1]):
                            st.success("Запис видалено!")
                            time.sleep(2)
                            st.experimental_rerun()
                
                if show_restore:
                    if st.button(f"♻️ Відновити", key=f"rest_{id}_{db_name}"):
                        if restore_record(id, db_name.split('_')[1]):
                            st.success("Запис відновлено!")
                            time.sleep(2)
                            st.experimental_rerun()
            
            st.markdown(f"<div style='font-size: 0.8rem; color: #777; margin-top: 10px;'>"
                      f"Додано: {timestamp.split('.')[0] if isinstance(timestamp, str) else timestamp}"
                      f"</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Помилка відображення запису: {str(e)}")

if __name__ == "__main__":
    main()