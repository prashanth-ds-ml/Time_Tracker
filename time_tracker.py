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
st.set_page_config(page_title="Focus Timer", layout="wide", initial_sidebar_state="collapsed")
POMODORO_MIN = 25
BREAK_MIN = 5
IST = pytz.timezone('Asia/Kolkata')
MONGO_URI = st.secrets["mongo_uri"]
DB_NAME = "time_tracker_db"
COLLECTION_NAME = "logs"
SOUND_PATH = "https://github.com/prashanth-ds-ml/Time_Tracker/raw/refs/heads/main/one_piece_overtake.mp3"

# === MongoDB Connection ===
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
users_collection = db["users"]

# === SOUND ALERT ===
def sound_alert():
    st.components.v1.html(f"""
        <audio id="alertAudio" autoplay>
            <source src="{SOUND_PATH}" type="audio/mpeg">
        </audio>
        <script>
            const playAudio = () => {{
                const audio = new Audio('{SOUND_PATH}');
                audio.volume = 0.6;
                audio.play().catch(err => console.log("Audio blocked:", err));
            }}
            setTimeout(playAudio, 500);
        </script>
    """, height=0)

# === USER MANAGEMENT ===
def get_all_users():
    return [u["username"] for u in users_collection.find({}, {"_id": 0, "username": 1})]

def add_user(username):
    if not users_collection.find_one({"username": username}):
        users_collection.insert_one({"username": username, "created_at": datetime.utcnow()})

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
    st.session_state.custom_categories = ["Learning", "Development", "Research", "Planning"]
if "user" not in st.session_state:
    st.session_state.user = None

# === USER SELECTION (TOP BAR) ===
users = get_all_users()
if not users:
    add_user("prashanth")
    users = ["prashanth"]

# Top navigation using columns
col_user, col_page, col_add = st.columns([2, 3, 2])

with col_user:
    # Ensure user selection is consistent
    if st.session_state.user not in users:
        st.session_state.user = users[0]
    
    current_index = users.index(st.session_state.user) if st.session_state.user in users else 0
    user_select = st.selectbox("👤 User", users, index=current_index, key="user_select")
    
    # Only update if actually changed to prevent unnecessary reruns
    if user_select != st.session_state.user:
        st.session_state.user = user_select
        st.rerun()

with col_page:
    page = st.selectbox("📍 Navigate", ["🎯 Focus Timer", "📝 Notes Saver", "📊 Analytics", "🗂️ Notes Viewer"])

with col_add:
    with st.expander("➕ Add User"):
        new_user = st.text_input("Username", placeholder="Enter new username")
        if st.button("Add") and new_user:
            if new_user.strip() and new_user not in users:
                add_user(new_user.strip())
                st.session_state.user = new_user.strip()
                st.rerun()
            elif new_user in users:
                st.warning("User already exists!")

# === NOTES FUNCTIONALITY ===
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
    st.success("✅ Note saved!")

# === PAGE ROUTING ===
if page == "🗂️ Notes Viewer":
    st.header("🗂️ Notes Viewer")

    col1, col2 = st.columns(2)
    with col1:
        note_start = st.date_input("📅 From", datetime.now(IST) - timedelta(days=7))
    with col2:
        note_end = st.date_input("📅 To", datetime.now(IST))

    notes_query = {
        "type": "Note",
        "user": st.session_state.user,
        "date": {"$gte": note_start.isoformat(), "$lte": note_end.isoformat()}
    }
    notes = list(collection.find(notes_query))

    if notes:
        for note in notes:
            with st.container():
                st.subheader(f"📅 {note['date']}")
                st.write(note['content'])
                st.divider()
    else:
        st.info("📭 No notes found in this date range")
    st.stop()

elif page == "📝 Notes Saver":
    st.header("📝 Daily Notes")

    with st.form("note_form", clear_on_submit=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            note_date = st.date_input("📅 Date", datetime.now(IST))
        with col2:
            note_content = st.text_area("✍️ Your thoughts...", placeholder="What did you learn today?", height=150)

        if st.form_submit_button("💾 Save Note", use_container_width=True):
            if note_content.strip():
                add_note(note_content.strip(), note_date.isoformat())
            else:
                st.warning("⚠️ Please add some content")
    st.stop()

# === MAIN FOCUS TIMER PAGE ===
if page == "🎯 Focus Timer":
    # Load and prepare data
    records = list(collection.find({"type": "Pomodoro", "user": st.session_state.user}))

    if records:
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date"], inplace=True)
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0).astype(int)

        today = datetime.now(IST).date()
        df_work = df[df["pomodoro_type"] == "Work"]
        work_today = df_work[df_work["date"].dt.date == today]

        # Progressive goals
        active_days = len(df_work.groupby(df_work["date"].dt.date).size())

        def get_adaptive_goal(days_active):
            if days_active <= 5:
                return 1, "🌱 Building", "Start small"
            elif days_active <= 12:
                return 2, "🔥 Growing", "Build momentum"
            elif days_active <= 19:
                return 3, "💪 Strong", "Push limits"
            else:
                return 4, "🚀 Peak", "Excellence"

        adaptive_goal, phase_name, phase_desc = get_adaptive_goal(active_days)
        today_progress = len(work_today)
        today_minutes = work_today['duration'].sum()
        progress_pct = min(100, (today_progress / adaptive_goal) * 100)
    else:
        active_days, adaptive_goal, today_progress, today_minutes, progress_pct = 0, 1, 0, 0, 0
        phase_name, phase_desc = "🚀 Start", "Begin journey"

    # === DAILY GOAL HEADER ===
    st.header("🎯 Today's Goal")

    # Goal metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📈 Phase", phase_name)

    with col2:
        st.metric("🎯 Progress", f"{today_progress}/{adaptive_goal}")

    with col3:
        st.metric("⏱️ Minutes Today", today_minutes)

    with col4:
        st.metric("📅 Active Days", active_days)

    # Progress bar
    st.progress(progress_pct / 100, text=f"{progress_pct:.0f}% complete - {phase_desc}")

    # Goal completion status
    if today_progress >= adaptive_goal:
        st.success("🎉 Daily goal achieved! Great work!")
    else:
        remaining = adaptive_goal - today_progress
        st.info(f"🎯 {remaining} more session{'s' if remaining != 1 else ''} to reach today's goal")

    st.divider()

    # === TIMER SECTION ===
    if st.session_state.start_time:
        duration = BREAK_MIN * 60 if st.session_state.is_break else POMODORO_MIN * 60
        remaining = int(st.session_state.start_time + duration - time.time())

        if remaining > 0:
            mins, secs = divmod(remaining, 60)
            session_type = "Break Time" if st.session_state.is_break else f"Working on: {st.session_state.task}"

            st.subheader(f"{'🧘' if st.session_state.is_break else '💼'} {session_type}")

            # Large timer display
            st.markdown(f"### ⏱️ {mins:02d}:{secs:02d}")

            # Motivational message
            if st.session_state.is_break:
                st.info("Take a breather! You're doing great 😊")
            else:
                st.info("Stay focused! You've got this 💪")

            time.sleep(1)
            st.rerun()
        else:
            # Session completed
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

            st.success("🎉 Session Complete!")
            if st.session_state.is_break:
                st.info("Break finished! Ready to get back to work?")
            else:
                st.info(f"Great work on: {st.session_state.task}")

            # Reset session
            st.session_state.task = ""
            st.session_state.category = ""
            st.session_state.start_time = None
            st.session_state.is_break = False

    # === QUICK START SECTION ===
    st.subheader("🚀 Quick Start")

    # Form for starting sessions
    col1, col2 = st.columns([1, 2])

    with col1:
        # Category selection with quick add
        cat_options = st.session_state.custom_categories + ["+ Add New"]
        category_select = st.selectbox("📂 Category", cat_options, key="cat_select")

        if category_select == "+ Add New":
            new_cat = st.text_input("New category", placeholder="e.g., Marketing")
            if new_cat and st.button("✅ Add Category", key="add_cat"):
                if new_cat not in st.session_state.custom_categories:
                    st.session_state.custom_categories.append(new_cat)
                    st.session_state.category = new_cat
                    st.rerun()
            st.session_state.category = new_cat if new_cat else ""
        else:
            st.session_state.category = category_select

    with col2:
        st.session_state.task = st.text_input("🎯 Task", placeholder="What are you working on?", key="task_input")

    # Action buttons
    col_work, col_break = st.columns(2)

    with col_work:
        if st.button("▶️ Start Work (25min)", use_container_width=True, type="primary"):
            if st.session_state.task.strip():
                st.session_state.start_time = time.time()
                st.session_state.is_break = False
                st.rerun()
            else:
                st.error("⚠️ Please enter a task")

    with col_break:
        if st.button("☕ Break (5min)", use_container_width=True):
            st.session_state.start_time = time.time()
            st.session_state.is_break = True
            st.session_state.category = ""
            st.session_state.task = ""
            st.rerun()

    # === TODAY'S SUMMARY ===
    if records:
        st.divider()
        st.subheader("📊 Today's Summary")

        # Quick stats
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("🎯 Work Sessions", today_progress)
        with col2:
            st.metric("⏱️ Focus Minutes", today_minutes)
        with col3:
            breaks_today = len(df[(df["date"].dt.date == today) & (df["pomodoro_type"] == "Break")])
            st.metric("☕ Breaks Taken", breaks_today)
        with col4:
            if today_progress >= adaptive_goal:
                st.metric("✅ Goal Status", "Achieved!")
            else:
                remaining_sessions = adaptive_goal - today_progress
                st.metric("🎯 Sessions Left", remaining_sessions)

# === ANALYTICS PAGE ===
elif page == "📊 Analytics":
    st.header("📊 Analytics Dashboard")

    records = list(collection.find({"type": "Pomodoro", "user": st.session_state.user}))

    if not records:
        st.info("📈 Analytics will appear after your first session")
        st.stop()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0).astype(int)

    df_work = df[df["pomodoro_type"] == "Work"]
    today = datetime.now(IST).date()

    # === KEY METRICS ===
    st.subheader("📈 Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_sessions = len(df_work)
        st.metric("🎯 Total Sessions", total_sessions)

    with col2:
        total_hours = df_work['duration'].sum() // 60
        st.metric("⏱️ Total Hours", total_hours)

    with col3:
        active_days = len(df_work.groupby(df_work["date"].dt.date).size())
        st.metric("📅 Active Days", active_days)

    with col4:
        if len(df_work) > 0:
            avg_daily = df_work.groupby(df_work["date"].dt.date).size().mean()
            st.metric("📊 Avg Daily", f"{avg_daily:.1f}")

    st.divider()

    # === VISUALIZATIONS ===

    # 1. Daily Performance (Last 30 days)
    st.subheader("📈 Daily Performance (Last 30 Days)")

    daily_data = []
    for i in range(30):
        date_check = today - timedelta(days=29-i)
        daily_work = df_work[df_work["date"].dt.date == date_check]
        daily_data.append({
            'date': date_check.strftime('%m/%d'),
            'sessions': len(daily_work),
            'minutes': daily_work['duration'].sum()
        })

    daily_df = pd.DataFrame(daily_data)

    if daily_df['minutes'].sum() > 0:
        fig = px.bar(daily_df, x='date', y='minutes', 
                    title="Daily Focus Minutes",
                    color='minutes', color_continuous_scale='Blues')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # 2. Category and Task Breakdown
    if len(df_work) > 0:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📂 Category Breakdown")
            category_data = df_work.groupby('category')['duration'].sum().sort_values(ascending=False)

            if len(category_data) > 0:
                fig = px.pie(values=category_data.values, names=category_data.index,
                           title="Time by Category")
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🎯 Top Tasks")
            task_data = df_work.groupby('task')['duration'].sum().sort_values(ascending=False).head(8)

            if len(task_data) > 0:
                fig = px.bar(x=task_data.values, y=task_data.index, 
                           orientation='h', title="Top Tasks by Time")
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

        # 3. Category-Task Bubble Chart
        st.subheader("🫧 Category-Task Bubble View")
        
        # Prepare data for bubble chart
        task_category_data = df_work.groupby(['category', 'task']).agg({
            'duration': 'sum',
            'date': 'count'
        }).rename(columns={'date': 'sessions'}).reset_index()
        
        if len(task_category_data) > 0:
            try:
                # Create bubble chart
                fig = px.scatter(task_category_data, 
                               x='category', 
                               y='task',
                               size='duration',
                               color='category',
                               hover_data=['sessions', 'duration'],
                               title="Task Distribution by Category (Bubble Size = Time Spent)",
                               labels={'duration': 'Minutes', 'sessions': 'Sessions'})
                
                fig.update_layout(
                    height=500,
                    xaxis_tickangle=-45,
                    showlegend=False
                )
                
                # Adjust bubble sizes more safely
                fig.update_traces(marker_sizemin=8, marker_sizemax=40)
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                # Fallback: Simple bar chart if bubble chart fails
                st.warning("Bubble chart unavailable, showing bar chart instead")
                fig = px.bar(task_category_data.sort_values('duration', ascending=False).head(10), 
                           x='duration', y='task', orientation='h',
                           title="Top Tasks by Duration")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Show summary table below
            with st.expander("📋 Detailed Breakdown"):
                summary_table = task_category_data.sort_values('duration', ascending=False)
                summary_table['duration_hours'] = (summary_table['duration'] / 60).round(2)
                st.dataframe(
                    summary_table[['category', 'task', 'sessions', 'duration_hours']],
                    column_config={
                        'category': 'Category',
                        'task': 'Task',
                        'sessions': 'Sessions',
                        'duration_hours': 'Hours'
                    },
                    use_container_width=True
                )

    st.divider()

    # === STREAK INFO ===
    st.subheader("🔥 Consistency Tracking")

    # Calculate current streak (minimum 2 sessions per day after day 12)
    daily_counts = df_work.groupby(df_work["date"].dt.date).size()
    current_streak = 0
    min_sessions = 1 if active_days <= 12 else 2

    for i in range(365):
        check_date = today - timedelta(days=i)
        day_count = daily_counts.get(check_date, 0)
        if day_count >= min_sessions:
            if i == 0:
                current_streak += 1
            else:
                current_streak += 1
        else:
            break

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🔥 Current Streak", f"{current_streak} days")

    with col2:
        max_streak = 0
        temp_streak = 0
        for i in range(365):
            check_date = today - timedelta(days=i)
            day_count = daily_counts.get(check_date, 0)
            if day_count >= min_sessions:
                temp_streak += 1
                max_streak = max(max_streak, temp_streak)
            else:
                temp_streak = 0
        st.metric("🏆 Best Streak", f"{max_streak} days")

    with col3:
        consistency = len([d for d in daily_counts.tail(7) if d >= min_sessions]) / 7 * 100
        st.metric("📊 Weekly Consistency", f"{consistency:.0f}%")

    # Streak explanation
    if active_days <= 12:
        st.info("💡 Currently in building phase: 1 session per day maintains your streak")
    else:
        st.info("💡 Growth phase: 2+ sessions per day needed to maintain streak")