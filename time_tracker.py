import streamlit as st
import time
import csv
import os
from datetime import datetime, timedelta
import pandas as pd
import pytz

# === CONFIG ===
POMODORO_MIN = 25
BREAK_MIN = 5
CSV_FILE = "pomodoro_log.csv"
SOUND_PATH = "sanji.mp3"  # Must be placed in the same directory or served statically
IST = pytz.timezone('Asia/Kolkata')
EXPECTED_COLS = ["Date", "Time", "Category", "Task", "Type", "Duration"]

# === SOUND ALERT (via HTML) ===
def sound_alert():
    st.components.v1.html(f"""
        <audio autoplay>
            <source src="{SOUND_PATH}" type="audio/mpeg">
        </audio>
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
                    # Handle old rows with fewer fields
                    date, time_str, task, task_type = row
                    category = "" if task_type == "Break" else "Misc"
                    duration = BREAK_MIN if task_type == "Break" else POMODORO_MIN
                    raw.append([date, time_str, category, task, task_type, duration])

        df = pd.DataFrame(raw, columns=EXPECTED_COLS)
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
        df.dropna(subset=["Date"], inplace=True)
        df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce").fillna(0).astype(int)

        # === Today Summary ===
        today = datetime.now(IST).date()
        df_today = df[df["Date"].dt.date == today]
        work_today = df_today[df_today["Type"] == "Work"]
        break_today = df_today[df_today["Type"] == "Break"]

        col1, col2, col3 = st.columns(3)
        col1.metric("💼 Work Today", f"{work_today['Duration'].sum()} min")
        col2.metric("☕ Break Today", f"{break_today['Duration'].sum()} min")
        col3.metric("🔁 Break Sessions", len(break_today))

        # === Daily Chart ===
        st.subheader("📆 Daily Work Summary")
        df_work = df[df["Type"] == "Work"]
        daily_sum = df_work.groupby(df["Date"].dt.date)["Duration"].sum()
        st.bar_chart(daily_sum)

        # === Category + Task Breakdown ===
        st.subheader("🧠 Time per Task in Each Category")
        cat_task = df_work.groupby(["Category", "Task"])["Duration"].sum().sort_values(ascending=False)
        st.dataframe(cat_task.reset_index().rename(columns={"Duration": "Minutes"}))

        # === Total Time Overall ===
        st.markdown("---")
        st.header("🧮 Overall Summary")
        total_min = df_work["Duration"].sum()
        st.write(f"**Total Work Time:** {total_min} min ({total_min//60} hr {total_min%60} min)")

        # === Most Productive Day ===
        df_cycles = df_work.groupby(df["Date"].dt.date).size() // 4
        if not df_cycles.empty:
            best_day = df_cycles.idxmax()
            st.write(f"**Most Productive Day:** {best_day} with {df_cycles.max()} Pomodoro cycle(s)")

        # === Streak Tracker ===
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
        st.error(f"Error processing CSV: {e}")
else:
    st.info("No log file yet. Start a Pomodoro to begin tracking.")

