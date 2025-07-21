
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

# === MODERN CSS STYLING ===
st.markdown("""
<style>
    /* Clean background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    
    /* Custom card styling */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #6366f1;
        margin-bottom: 15px;
    }
    
    /* Timer display */
    .timer-display {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
    }
    
    /* Goal progress bar */
    .progress-container {
        background: #f0f0f0;
        border-radius: 25px;
        height: 25px;
        margin: 15px 0;
        overflow: hidden;
    }
    
    /* Clean section headers */
    .section-header {
        background: white;
        padding: 15px 20px;
        border-radius: 10px;
        margin: 20px 0 10px 0;
        box-shadow: 0 1px 5px rgba(0,0,0,0.1);
        border-left: 3px solid #10b981;
    }
    
    /* Compact input styling */
    .stSelectbox, .stTextInput {
        margin-bottom: 10px;
    }
    
    /* Clean button styling */
    .stButton button {
        width: 100%;
        border-radius: 8px;
        border: none;
        padding: 12px;
        font-weight: 600;
    }
    
    /* Stats cards */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

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

# Clean top navigation
col_user, col_page, col_add = st.columns([2, 3, 2])

with col_user:
    user_select = st.selectbox("👤 User", users, 
                              index=users.index(st.session_state.user) if st.session_state.user in users else 0,
                              key="user_select")
    st.session_state.user = user_select

with col_page:
    page = st.selectbox("📍 Navigate", ["🎯 Focus Timer", "📝 Notes Saver", "📊 Analytics", "🗂️ Notes Viewer"])

with col_add:
    with st.expander("➕ Add User"):
        new_user = st.text_input("Username", placeholder="Enter new username")
        if st.button("Add") and new_user:
            if new_user not in users:
                add_user(new_user)
                st.session_state.user = new_user
                st.rerun()

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
    st.markdown('<div class="section-header"><h2>🗂️ Notes Viewer</h2></div>', unsafe_allow_html=True)
    
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
            st.markdown(f"""
            <div class="metric-card">
                <h4>📅 {note['date']}</h4>
                <p>{note['content']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📭 No notes found in this date range")
    st.stop()

elif page == "📝 Notes Saver":
    st.markdown('<div class="section-header"><h2>📝 Daily Notes</h2></div>', unsafe_allow_html=True)
    
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
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 25px; border-radius: 15px; margin-bottom: 25px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; font-size: 28px;">🎯 Today's Goal</h1>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">{phase_name} • {phase_desc}</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 36px; font-weight: bold;">{today_progress}/{adaptive_goal}</div>
                <div style="font-size: 14px; opacity: 0.8;">Sessions</div>
            </div>
        </div>
        <div class="progress-container" style="background: rgba(255,255,255,0.2); margin-top: 15px;">
            <div style="background: rgba(255,255,255,0.9); width: {progress_pct}%; height: 100%; 
                        border-radius: 25px; transition: width 0.3s ease;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 10px; font-size: 14px;">
            <span>⏱️ {today_minutes} minutes today</span>
            <span>📅 Day {active_days} of journey</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # === TIMER SECTION ===
    if st.session_state.start_time:
        duration = BREAK_MIN * 60 if st.session_state.is_break else POMODORO_MIN * 60
        remaining = int(st.session_state.start_time + duration - time.time())
        
        if remaining > 0:
            mins, secs = divmod(remaining, 60)
            session_type = "Break Time" if st.session_state.is_break else f"Working: {st.session_state.task}"
            
            st.markdown(f"""
            <div class="timer-display">
                <h2 style="margin: 0; font-size: 24px;">{'🧘' if st.session_state.is_break else '💼'} {session_type}</h2>
                <div style="font-size: 48px; font-weight: bold; margin: 20px 0;">{mins:02d}:{secs:02d}</div>
                <div style="opacity: 0.8;">Stay focused! You've got this 💪</div>
            </div>
            """, unsafe_allow_html=True)
            
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
            
            st.markdown(f"""
            <div style="background: #10b981; color: white; padding: 20px; border-radius: 10px; text-align: center;">
                <h2>🎉 Session Complete!</h2>
                <p>{'Break finished!' if st.session_state.is_break else 'Great work on: ' + st.session_state.task}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Reset session
            st.session_state.task = ""
            st.session_state.category = ""
            st.session_state.start_time = None
            st.session_state.is_break = False

    # === QUICK START SECTION ===
    st.markdown('<div class="section-header"><h3>🚀 Quick Start</h3></div>', unsafe_allow_html=True)
    
    # Compact form
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Category selection with quick add
        cat_options = st.session_state.custom_categories + ["+ Add New"]
        category_select = st.selectbox("📂 Category", cat_options, key="cat_select")
        
        if category_select == "+ Add New":
            new_cat = st.text_input("New category", placeholder="e.g., Marketing")
            if new_cat and st.button("✅ Add", key="add_cat"):
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
        st.markdown('<div class="section-header"><h3>📊 Today\'s Summary</h3></div>', unsafe_allow_html=True)
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Sessions", today_progress)
        with col2:
            st.metric("⏱️ Minutes", today_minutes)
        with col3:
            breaks_today = len(df[(df["date"].dt.date == today) & (df["pomodoro_type"] == "Break")])
            st.metric("☕ Breaks", breaks_today)
        with col4:
            if today_progress >= adaptive_goal:
                st.metric("🎉 Status", "Goal Met!")
            else:
                remaining_sessions = adaptive_goal - today_progress
                st.metric("🎯 To Goal", f"{remaining_sessions} left")

# === ANALYTICS PAGE ===
elif page == "📊 Analytics":
    st.markdown('<div class="section-header"><h2>📊 Analytics Dashboard</h2></div>', unsafe_allow_html=True)
    
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
    
    # === VISUALIZATIONS ===
    
    # 1. Daily Performance (Last 30 days)
    st.subheader("📈 Daily Performance")
    
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
                    title="Daily Focus Minutes (Last 30 Days)",
                    color='minutes', color_continuous_scale='Blues')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # 2. Category Breakdown
    if len(df_work) > 0:
        col1, col2 = st.columns(2)
        
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
            task_data = df_work.groupby('task')['duration'].sum().sort_values(ascending=False).head(10)
            
            if len(task_data) > 0:
                fig = px.bar(x=task_data.values, y=task_data.index, 
                           orientation='h', title="Top Tasks by Time")
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
    
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

st.markdown("<br><br>", unsafe_allow_html=True)
