import streamlit as st
import time
import csv
import os
from datetime import datetime, timedelta
import pandas as pd
import pytz
import plotly.express as px
from pymongo import MongoClient
import hashlib

# === CONFIG ===
st.set_page_config(page_title="Pomodoro Tracker", layout="centered")  # Must be first Streamlit command
POMODORO_MIN = 25
BREAK_MIN = 5
IST = pytz.timezone('Asia/Kolkata')
MONGO_URI = st.secrets["mongo_uri"]
DB_NAME = "time_tracker_db"
COLLECTION_NAME = "logs"
SOUND_PATH = "https://github.com/prashanth-ds-ml/Time_Tracker/raw/refs/heads/main/one_piece_overtake.mp3"
BACKGROUND_IMAGE = "https://github.com/prashanth-ds-ml/Time_Tracker/blob/main/Roronoa%20zoro.jpeg?raw=true"

# === CUSTOM CSS FOR BACKGROUND ===
st.markdown(f"""
    <style>
        .stApp {{
            background-image: url('{BACKGROUND_IMAGE}');
            background-size: contain;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
    </style>
""", unsafe_allow_html=True)

# === SOUND ALERT with JS + HTML fallback ===
def sound_alert():
    st.components.v1.html(f"""
        <audio id="alertAudio" autoplay>
            <source src="{SOUND_PATH}" type="audio/mpeg">
            Your browser does not support the audio element.
        </audio>
        <script>
            const playAudio = () => {{
                const audio = new Audio('{SOUND_PATH}');
                audio.volume = 0.8;
                audio.play().catch(err => console.log("Autoplay issue:", err));
            }}
            setTimeout(playAudio, 1000);
        </script>
    """, height=0)

# === SESSION STATE INIT ===
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "is_break" not in st.session_state:
    st.session_state.is_break = False
if "category" not in st.session_state:
    st.session_state.category = ""
if "task" not in st.session_state:
    st.session_state.task = ""
if "custom_categories" not in st.session_state:
    st.session_state.custom_categories = ["Learning", "Startup"]

# === MongoDB Connection ===
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# === FUNCTIONS ===
def add_note(content, date, category="", task=""):
    note_id = hashlib.sha256(f"{date}_{category}_{task}".encode("utf-8")).hexdigest()
    note_doc = {
        "_id": note_id,
        "type": "Note",
        "date": date,
        "Notes": content,
        "category": category,
        "task": task,
        "created_at": datetime.utcnow()
    }
    collection.update_one({"_id": note_id}, {"$set": note_doc}, upsert=True)
    st.success("Note saved successfully!")

# === SIDEBAR NAVIGATION ===
st.sidebar.title("📁 Pages")
page = st.sidebar.radio("Go to", ["Pomodoro Tracker", "Notes Viewer", "Notes Saver"])

if page == "Notes Viewer":
    st.title("🗂️ Notes Viewer")
    note_start = st.date_input("From", datetime.now(IST) - timedelta(days=7))
    note_end = st.date_input("To", datetime.now(IST))
    notes_query = {
        "type": "Note",
        "date": {"$gte": note_start.isoformat(), "$lte": note_end.isoformat()}
    }
    notes = list(collection.find(notes_query))
    if notes:
        for note in notes:
            st.markdown(f"**{note['date']}**")
            st.markdown(f"*{note['category']} - {note['task']}*")
            st.markdown(note['Notes'])
            st.markdown("---")
    else:
        st.info("No notes in this range.")
    st.stop()

elif page == "Notes Saver":
    st.title("📝 Save Daily Note")
    with st.form("add_note"):
        note_date = st.date_input("Date", datetime.now(IST))
        cat_options = st.session_state.custom_categories + ["➕ Add New Category"]
        category_selection = st.selectbox("Category", cat_options)
        if category_selection == "➕ Add New Category":
            new_cat = st.text_input("Enter New Category")
            if new_cat:
                if new_cat not in st.session_state.custom_categories:
                    st.session_state.custom_categories.append(new_cat)
                note_category = new_cat
        else:
            note_category = category_selection
        note_task = st.text_input("Task")
        note_content = st.text_area("Note Content")
        submitted = st.form_submit_button("💾 Save Note")
        if submitted:
            add_note(note_content, note_date.isoformat(), note_category, note_task)
    st.stop()

# === UI ===
# ... (Rest of Pomodoro Tracker remains unchanged)
