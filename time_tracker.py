import streamlit as st
import time
import csv
import os
from datetime import datetime, timedelta
import pandas as pd
import pygame
import pytz

# === CONFIG ===
POMODORO_MIN = 25
BREAK_MIN = 5
CSV_FILE = "pomodoro_log.csv"
SOUND_PATH = "sanji.mp3"  # Ensure this file exists in your folder
IST = pytz.timezone('Asia/Kolkata')
EXPECTED_COLS = ["Date", "Time", "Category", "Task", "Type", "Duration"]

# === SOUND ALERT ===
def play_alert(path):
    try:
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
    except Exception as e:
        st.warning(f"Sound alert failed: {e}")

# === SESSION STATE ===
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

# === UI SETUP ===
st.set_page_config(page_title="Pomodoro Tracker", layout="centered")
st.title("⏱️ Time Tracker (IST)")
st.markdown("Track focused work with custom categories, alerts, and visual summaries.")

st.markdown("---")
st.header("🎯 Start a Work Session")

# === CATEGORY SELECTION ===
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

# === TASK INPUT ===
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

# === TIMER ===
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
        log_entry = [
            now_ist.strftime("%d-%m-%Y"),
            now_ist.strftime("%I:%M %p"),
            st.session_state.category if not st.session_state.is_break else "",
            st.session_state.task if not st.session_state.is_break else "",
            "Break" if st.session_state.is_break else "Work",
            BREAK_MIN if st.session_state.is_break else POMODORO_MIN
        ]

        file_exists = os.path.exists(CSV_FILE)
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(EXPECTED_COLS)
            writer.writerow(log_entry)

        play_alert(SOUND_PATH)
        st.balloons()
        st.success(f"{'Break' if st.session_state.is_break else 'Pomodoro'} session completed!")

        st.session_state.task = ""
        st.session_state.category = ""
        st.session_state.start_time = None
        st.session_state.is_break = False

# === ANALYTICS ===
st.markdown("---")
st.header("📊 Daily Focus Summary")

if os.path.exists(CSV_FILE):
    try:
        raw = []
        with open(CSV_FILE, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if len(row) == 6:
                    raw.append(row)
                elif len(row) == 4:
                    # Older format fallback
                    row = row[:2] + ["", "", row[2], row[3]]
                    raw.append(row)
        df = pd.DataFrame(raw, columns=EXPECTED_COLS)
        df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y", errors='coerce')
        df.dropna(subset=["Date"], inplace=True)
        df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce").fillna(0).astype(int)

        today = datetime.now(IST).date()
        df_today = df[df["Date"].dt.date == today]
        work_today = df_today[df_today["Type"] == "Work"]
        break_today = df_today[df_today["Type"] == "Break"]

        col1, col2, col3 = st.columns(3)
        col1.metric("💼 Total Work Time", f"{work_today['Duration'].sum()} min")
        col2.metric("☕ Total Break Time", f"{break_today['Duration'].sum()} min")
        col3.metric("🧘 Break Sessions", f"{len(break_today)}")

        df_work = df[df["Type"] == "Work"]
        daily_minutes = df_work.groupby(df_work['Date'].dt.date)['Duration'].sum()
        st.markdown("#### 📈 Daily Work Chart")
        st.bar_chart(daily_minutes)

        cat_task_totals = df_work.groupby(["Category", "Task"])["Duration"].sum().sort_values(ascending=False)
        st.markdown("#### 🧠 Time Spent on Each Task per Category")
        st.dataframe(cat_task_totals.reset_index().rename(columns={"Duration": "Minutes"}))

        total_minutes = df_work["Duration"].sum()
        total_hours = total_minutes // 60
        st.markdown("---")
        st.header("🧮 All-Time Productive Stats")
        st.write(f"**Total Productive Time:** {total_minutes} min ({total_hours} hr {total_minutes % 60} min)")

        df_cycles = df_work.groupby(df_work['Date'].dt.date).size() // 4
        if not df_cycles.empty:
            st.write(f"**Most Productive Day:** {df_cycles.idxmax()} with {df_cycles.max()} Pomodoro cycle(s)")

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
    except Exception as e:
        st.error(f"Error reading or parsing CSV: {e}")
else:
    st.info("Start a Pomodoro session to begin tracking.")
