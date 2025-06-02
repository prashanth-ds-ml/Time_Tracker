import streamlit as st
import time
import hashlib
from datetime import datetime, timedelta
import pandas as pd
import pytz
import plotly.express as px
from pymongo import MongoClient

# === CONFIG ===
POMODORO_MIN = 25
BREAK_MIN = 5
SOUND_PATH = "https://github.com/prashanth-ds-ml/Time_Tracker/raw/refs/heads/main/sanji.mp3"
IST = pytz.timezone('Asia/Kolkata')
DB_NAME = "time_tracker_db"
COLLECTION_NAME = "logs"

# === MongoDB Connection ===
MONGO_URI = st.secrets["mongo_uri"]
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

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

# === UI ===
st.set_page_config(page_title="Pomodoro Tracker", layout="centered")
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

# === SOUND ALERT ===
def sound_alert():
    st.components.v1.html(f"""
        <audio id=\"alertAudio\" autoplay>
            <source src=\"{SOUND_PATH}\" type=\"audio/mpeg\">
            Your browser does not support the audio element.
        </audio>
    """, height=0)

# === TIMER LOGIC ===
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
        date_str = now_ist.strftime("%d-%m-%Y")
        time_str = now_ist.strftime("%I:%M %p")
        category = st.session_state.category if not st.session_state.is_break else ""
        task = st.session_state.task if not st.session_state.is_break else ""
        task_type = "Break" if st.session_state.is_break else "Work"
        duration = BREAK_MIN if st.session_state.is_break else POMODORO_MIN

        def generate_log_id():
            key = f"{date_str}_{time_str}_{category}_{task}_{task_type}"
            return hashlib.sha256(key.encode()).hexdigest()

        log_entry = {
            "_id": generate_log_id(),
            "type": "Pomodoro",
            "date": datetime.strptime(date_str, "%d-%m-%Y").date().isoformat(),
            "time": time_str,
            "category": category,
            "task": task,
            "session_type": task_type,
            "duration": duration,
            "created_at": datetime.utcnow()
        }

        collection.update_one({"_id": log_entry["_id"]}, {"$set": log_entry}, upsert=True)
        sound_alert()
        st.balloons()
        st.success(f"{task_type} session completed!")

        st.session_state.task = ""
        st.session_state.category = ""
        st.session_state.start_time = None
        st.session_state.is_break = False

# === ANALYTICS SECTION ===
st.markdown("---")
st.header("📊 Productivity Analytics")

mongo_logs = list(collection.find({"type": "Pomodoro"}))
if mongo_logs:
    df = pd.DataFrame(mongo_logs)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0).astype(int)
    today = datetime.now(IST).date()

    df_today = df[df["date"].dt.date == today]
    work_today = df_today[df_today["type"] == "Work"]
    break_today = df_today[df_today["type"] == "Break"]

    col1, col2, col3 = st.columns(3)
    col1.metric("💼 Work Today", f"{work_today['duration'].sum()} min")
    col2.metric("☕ Break Today", f"{break_today['duration'].sum()} min")
    col3.metric("🔁 Break Sessions", len(break_today))

    st.subheader("📆 Daily Work Summary")
    df_work = df[df["session_type"] == "Work"]
    daily_sum = df_work.groupby(df["date"].dt.date)["duration"].sum().reset_index()
    daily_sum['DateStr'] = daily_sum['date'].astype(str)
    fig = px.bar(daily_sum, x='DateStr', y="duration", title="Daily Work Duration", labels={"duration": "Minutes", "date": "Date"})
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
    st.info("No logs found in database. Start tracking to see analytics.")

# === NOTES SECTION ===
st.markdown("---")
st.header("📝 Add Daily Note")

with st.form("note_form"):
    note_date = st.date_input("Date", value=datetime.now().date())
    note_title = st.text_input("Title")
    note_content = st.text_area("Note")
    note_category = st.selectbox("Category", st.session_state.custom_categories)
    note_task = st.text_input("Task")
    submitted = st.form_submit_button("Save Note")

    if submitted and note_title and note_content:
        note_id = hashlib.sha256(f"{note_date}_{note_title}".encode()).hexdigest()
        note_doc = {
            "_id": note_id,
            "type": "Note",
            "date": note_date.isoformat(),
            "title": note_title,
            "content": note_content,
            "category": note_category,
            "task": note_task,
            "created_at": datetime.utcnow()
        }
        collection.update_one({"_id": note_id}, {"$set": note_doc}, upsert=True)
        st.success("Note saved!")
