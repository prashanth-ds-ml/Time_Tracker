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
BACKGROUND_IMAGE = "https://i.pinimg.com/736x/ef/d4/04/efd404ef0270e3ab0561177425626d4c.jpg"

# === MongoDB Connection ===
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
users_collection = db["users"]

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

# === MongoDB Connection ===
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
users_collection = db["users"]

# === USER MANAGEMENT ===
def get_all_users():
    return [u["username"] for u in users_collection.find({}, {"_id": 0, "username": 1})]

def add_user(username):
    if not users_collection.find_one({"username": username}):
        users_collection.insert_one({"username": username, "created_at": datetime.utcnow()})

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
if "user" not in st.session_state:
    st.session_state.user = None

# === SIDEBAR USER SELECTION ===
st.sidebar.title("👤 User Management")
users = get_all_users()
if not users:
    add_user("prashanth")
    users = ["prashanth"]

user_select = st.sidebar.selectbox("Select User", users, index=users.index(st.session_state.user) if st.session_state.user in users else 0)
st.session_state.user = user_select

with st.sidebar.expander("➕ Add New User"):
    new_user = st.text_input("New Username", key="new_user_input")
    if st.button("Add User"):
        if new_user and new_user not in users:
            add_user(new_user)
            st.session_state.user = new_user
            st.rerun()
        elif new_user in users:
            st.warning("User already exists.")

# === FUNCTIONS ===
def add_note(content, date):
    note_id = hashlib.sha256(f"{date}_{content}_{st.session_state.user}".encode("utf-8")).hexdigest()
    note_doc = {
        "_id": note_id,
        "type": "Note",
        "date": date,
        "content": content,
        "user": st.session_state.user,
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
        "user": st.session_state.user,
        "date": {"$gte": note_start.isoformat(), "$lte": note_end.isoformat()}
    }
    notes = list(collection.find(notes_query))
    if notes:
        for note in notes:
            st.markdown(f"**{note['date']}**")
            st.markdown(note['content'])
            st.markdown("---")
    else:
        st.info("No notes in this range.")
    st.stop()

elif page == "Notes Saver":
    st.title("📝 Save Daily Note")

    with st.form("add_note_form"):
        note_date = st.date_input("Date", datetime.now(IST))
        note_content = st.text_area("Note Content")
        submitted = st.form_submit_button("💾 Save Note")

    if submitted:
        if not note_content.strip():
            st.warning("Note content cannot be empty.")
        else:
            add_note(note_content.strip(), note_date.isoformat())

    st.stop()

# === UI ===
st.title("⏱️ Time Tracker (IST)")
st.markdown("Track focused work with custom categories, alerts, and visual summaries.")

st.markdown("---")
st.header("🎯 Start a Work Session")

# === Category Input ===
cat_options = st.session_state.custom_categories + ["➕ Add New Category"]
category_selection = st.selectbox("Select Category", cat_options)

if category_selection == "➕ Add New Category":
    new_cat = st.text_input("Enter New Category")
    if new_cat:
        if new_cat not in st.session_state.custom_categories:
            st.session_state.custom_categories.append(new_cat)
        st.session_state.category = new_cat
else:
    st.session_state.category = category_selection

# === Task Input ===
st.session_state.task = st.text_input("Enter Task (e.g., MongoDB, ESPnet)").strip()

col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Work Pomodoro (25 min)"):
        if st.session_state.task:
            st.session_state.start_time = time.time()
            st.session_state.is_break = False
            st.success(f"Started: {st.session_state.category} → {st.session_state.task}")
        else:
            st.warning("Please enter a task before starting.")
with col2:
    if st.button("☕ Start Break (5 min)"):
        st.session_state.category = ""
        st.session_state.task = ""
        st.session_state.start_time = time.time()
        st.session_state.is_break = True
        st.success("Break started!")

# === Timer Logic ===
if st.session_state.start_time:
    duration = BREAK_MIN * 60 if st.session_state.is_break else POMODORO_MIN * 60
    end_time = st.session_state.start_time + duration
    remaining = int(end_time - time.time())

    if remaining > 0:
        mins, secs = divmod(remaining, 60)
        st.markdown(f"### {'🧘 Break' if st.session_state.is_break else '💼 Working on: ' + st.session_state.task}")
        st.info(f"⏳ Time Left: {mins:02}:{secs:02}")
        time.sleep(1)
        st.rerun()
    else:
        now_ist = datetime.now(IST)
        doc = {
            "type": "Pomodoro",
            "date": now_ist.date().isoformat(),
            "time": now_ist.strftime("%I:%M %p"),
            "category": st.session_state.category if not st.session_state.is_break else "",
            "task": st.session_state.task if not st.session_state.is_break else "",
            "pomodoro_type": "Break" if st.session_state.is_break else "Work",
            "duration": BREAK_MIN if st.session_state.is_break else POMODORO_MIN,
            "user": st.session_state.user,
            "created_at": datetime.utcnow()
        }
        collection.insert_one(doc)
        sound_alert()
        st.balloons()
        st.success(f"{'Break' if st.session_state.is_break else 'Pomodoro'} session completed!")
        st.session_state.task = ""
        st.session_state.category = ""
        st.session_state.start_time = None
        st.session_state.is_break = False

# === ANALYTICS SECTION ===
st.markdown("---")
st.header("📊 Productivity Analytics")

# === Load Pomodoro Logs from MongoDB ===
records = list(collection.find({"type": "Pomodoro", "user": st.session_state.user}))
if records:
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0).astype(int)

    today = datetime.now(IST).date()
    df_today = df[df["date"].dt.date == today]
    work_today = df_today[df_today["pomodoro_type"] == "Work"]
    break_today = df_today[df_today["pomodoro_type"] == "Break"]

    col1, col2, col3 = st.columns(3)
    col1.metric("💼 Work Today", f"{work_today['duration'].sum()} min")
    col2.metric("☕ Break Today", f"{break_today['duration'].sum()} min")
    col3.metric("🔁 Break Sessions", len(break_today))

    st.subheader("📆 Daily Work Summary")
    df_work = df[df["pomodoro_type"] == "Work"]

    # Group by date and sum durations
    daily_sum = df_work.groupby(df_work["date"].dt.date)["duration"].sum().reset_index()
    daily_sum = daily_sum.sort_values("date")
    daily_sum["date_str"] = daily_sum["date"].apply(lambda x: x.strftime("%b %d %Y"))

    # Plot
    fig = px.bar(
        daily_sum,
        x="date_str",
        y="duration",
        title="Daily Work Duration",
        labels={"duration": "Minutes", "date_str": "Date"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧠 Time per Task in Each Category")
    cat_task = df_work.groupby(["category", "task"])["duration"].sum().sort_values(ascending=False)
    st.dataframe(cat_task.reset_index().rename(columns={"duration": "Minutes"}))

    st.markdown("---")
    st.header("🧮 Overall Summary")
    total_min = df_work["duration"].sum()
    st.write(f"**Total Work Time:** {total_min} min ({total_min//60} hr {total_min%60} min)")

    df_cycles = df_work.groupby(df["date"].dt.date).size() // 4
    if not df_cycles.empty:
        best_day = df_cycles.idxmax()
        st.write(f"**Most Productive Day:** {best_day} with {df_cycles.max()} Pomodoro cycle(s)")

    st.markdown("---")
    st.header("🔥 Streak Tracker (4+ Pomodoros/day)")

    streak = 0
    best_streak = 0
    current = 0
    for i in range(30):
        check_date = today - timedelta(days=i)
        if check_date in df_cycles.index and df_cycles[check_date] >= 1:
            current += 1
            best_streak = max(best_streak, current)
            if i == 0:
                streak = current
        else:
            if i == 0:
                streak = 0
            current = 0

    st.metric("🔥 Current Streak", f"{streak} day(s)")
    st.metric("🏆 Best Streak", f"{best_streak} day(s)")
else:
    st.info("No log records found in MongoDB.")

