
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
st.set_page_config(
    page_title="Focus Timer", 
    layout="wide", 
    initial_sidebar_state="collapsed",
    menu_items={'About': "Focus Timer - Track your productivity with Pomodoro technique"}
)

POMODORO_MIN = 25
BREAK_MIN = 5
IST = pytz.timezone('Asia/Kolkata')
SOUND_PATH = "https://github.com/prashanth-ds-ml/Time_Tracker/raw/refs/heads/main/one_piece_overtake.mp3"

# MongoDB Configuration
@st.cache_resource
def init_database():
    """Initialize database connection - cached for performance"""
    try:
        MONGO_URI = st.secrets["mongo_uri"]
        client = MongoClient(MONGO_URI)
        db = client["time_tracker_db"]
        return db
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        st.stop()

db = init_database()
collection = db["logs"]
users_collection = db["users"]

# === UTILITY FUNCTIONS ===
def sound_alert():
    """Play completion sound"""
    st.components.v1.html(f"""
        <audio autoplay><source src="{SOUND_PATH}" type="audio/mpeg"></audio>
        <script>
            const audio = new Audio('{SOUND_PATH}');
            audio.volume = 0.6;
            audio.play().catch(err => console.log("Audio blocked:", err));
        </script>
    """, height=0)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_user_data(username):
    """Get user's pomodoro data - cached for performance"""
    records = list(collection.find({"type": "Pomodoro", "user": username}))
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0).astype(int)
    return df

def get_adaptive_goal(active_days):
    """Calculate progressive goals based on habit formation science"""
    if active_days <= 5:
        return 1, "🌱 Building", "Start small - consistency over intensity"
    elif active_days <= 12:
        return 2, "🔥 Growing", "Building momentum - you're doing great!"
    elif active_days <= 19:
        return 3, "💪 Strong", "Push your limits - you're in the zone!"
    else:
        return 4, "🚀 Peak", "Excellence mode - maintain this peak!"

def save_pomodoro_session(category, task, is_break, duration, user):
    """Save completed session to database"""
    now_ist = datetime.now(IST)
    doc = {
        "type": "Pomodoro",
        "date": now_ist.date().isoformat(),
        "time": now_ist.strftime("%I:%M %p"),
        "category": category if not is_break else "",
        "task": task if not is_break else "",
        "pomodoro_type": "Break" if is_break else "Work",
        "duration": duration,
        "user": user,
        "created_at": datetime.utcnow()
    }
    collection.insert_one(doc)
    # Clear cache after new data
    get_user_data.clear()

# === SESSION STATE INITIALIZATION ===
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "start_time": None,
        "is_break": False,
        "category": "",
        "task": "",
        "custom_categories": ["Learning", "Development", "Research", "Planning"],
        "user": None,
        "page": "🎯 Focus Timer"
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# === USER MANAGEMENT ===
@st.cache_data(ttl=60)
def get_all_users():
    """Get all users - cached for performance"""
    return [u["username"] for u in users_collection.find({}, {"_id": 0, "username": 1})]

def add_user(username):
    """Add new user if doesn't exist"""
    if not users_collection.find_one({"username": username}):
        users_collection.insert_one({"username": username, "created_at": datetime.utcnow()})
        get_all_users.clear()  # Clear cache
        return True
    return False

# === HEADER NAVIGATION ===
def render_header():
    """Render top navigation bar"""
    users = get_all_users()
    if not users:
        add_user("prashanth")
        users = ["prashanth"]

    # Ensure valid user selection
    if st.session_state.user not in users:
        st.session_state.user = users[0]

    col_user, col_page, col_add = st.columns([2, 3, 2])

    with col_user:
        current_index = users.index(st.session_state.user) if st.session_state.user in users else 0
        selected_user = st.selectbox("👤 User", users, index=current_index, key="user_select")
        
        if selected_user != st.session_state.user:
            st.session_state.user = selected_user
            st.rerun()

    with col_page:
        pages = ["🎯 Focus Timer", "📝 Notes Saver", "📊 Analytics", "🗂️ Notes Viewer"]
        selected_page = st.selectbox("📍 Navigate", pages, 
                                   index=pages.index(st.session_state.page) if st.session_state.page in pages else 0)
        st.session_state.page = selected_page

    with col_add:
        with st.expander("➕ Add User"):
            new_user = st.text_input("Username", placeholder="Enter new username", key="new_user_input")
            if st.button("Add", key="add_user_btn") and new_user:
                if new_user.strip():
                    if add_user(new_user.strip()):
                        st.session_state.user = new_user.strip()
                        st.success("✅ User added!")
                        st.rerun()
                    else:
                        st.warning("User already exists!")

# === TIMER LOGIC ===
def render_timer():
    """Render active timer display"""
    if not st.session_state.start_time:
        return False
        
    duration = BREAK_MIN * 60 if st.session_state.is_break else POMODORO_MIN * 60
    remaining = int(st.session_state.start_time + duration - time.time())

    if remaining > 0:
        mins, secs = divmod(remaining, 60)
        session_type = "Break Time" if st.session_state.is_break else f"Working on: {st.session_state.task}"

        # Timer display
        st.subheader(f"{'🧘' if st.session_state.is_break else '💼'} {session_type}")
        
        # Large countdown
        timer_col1, timer_col2, timer_col3 = st.columns([1, 2, 1])
        with timer_col2:
            st.markdown(f"<h1 style='text-align: center; font-size: 4rem;'>⏱️ {mins:02d}:{secs:02d}</h1>", 
                       unsafe_allow_html=True)

        # Progress bar
        progress = 1 - (remaining / duration)
        st.progress(progress)

        # Motivational message
        if st.session_state.is_break:
            st.info("🧘 Take a breather! You're doing great")
        else:
            st.info("💪 Stay focused! You've got this")

        time.sleep(1)
        st.rerun()
        return True
    else:
        # Session completed
        duration_min = BREAK_MIN if st.session_state.is_break else POMODORO_MIN
        save_pomodoro_session(
            st.session_state.category, 
            st.session_state.task, 
            st.session_state.is_break, 
            duration_min,
            st.session_state.user
        )
        
        sound_alert()
        st.balloons()
        st.success("🎉 Session Complete!")
        
        if st.session_state.is_break:
            st.info("Break finished! Ready to get back to work?")
        else:
            st.info(f"Great work on: {st.session_state.task}")

        # Reset session
        st.session_state.start_time = None
        st.session_state.is_break = False
        st.session_state.task = ""
        st.session_state.category = ""
        return True

# === DAILY TARGET PLANNER ===
def save_daily_target(target, user):
    """Save user's daily target to database"""
    today = datetime.now(IST).date().isoformat()
    target_doc = {
        "type": "DailyTarget",
        "date": today,
        "target": target,
        "user": user,
        "created_at": datetime.utcnow()
    }
    collection.update_one(
        {"type": "DailyTarget", "date": today, "user": user},
        {"$set": target_doc},
        upsert=True
    )

def get_daily_target(user):
    """Get user's daily target for today"""
    today = datetime.now(IST).date().isoformat()
    target_doc = collection.find_one({
        "type": "DailyTarget", 
        "date": today, 
        "user": user
    })
    return target_doc["target"] if target_doc else None

def render_daily_target_planner(df, today_progress):
    """Render daily target planning interface"""
    st.markdown("## 🎯 Daily Target Planner")
    
    # Get existing target or adaptive suggestion
    current_target = get_daily_target(st.session_state.user)
    
    if df.empty:
        active_days = 0
        suggested_target, phase_name, phase_desc = 1, "🌱 Building", "Start small - consistency over intensity"
    else:
        active_days = len(df[df["pomodoro_type"] == "Work"].groupby(df["date"].dt.date).size())
        suggested_target, phase_name, phase_desc = get_adaptive_goal(active_days)

    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Target setting interface
        st.markdown("### 📋 Set Your Target")
        
        # Show current target or let user set one
        if current_target:
            st.info(f"✅ Today's target: **{current_target} Pomodoros**")
            
            # Option to change target
            with st.expander("🔄 Change Today's Target"):
                new_target = st.number_input(
                    "New target", 
                    min_value=1, 
                    max_value=12, 
                    value=current_target,
                    key="change_target"
                )
                if st.button("💾 Update Target", key="update_target_btn"):
                    save_daily_target(new_target, st.session_state.user)
                    st.success("🎯 Target updated!")
                    st.rerun()
        else:
            st.markdown(f"💡 **Suggested:** {suggested_target} Pomodoros ({phase_name})")
            target_input = st.number_input(
                "How many Pomodoros today?", 
                min_value=1, 
                max_value=12, 
                value=suggested_target,
                key="daily_target_input"
            )
            
            if st.button("🎯 Set Daily Target", key="set_target_btn", use_container_width=True):
                save_daily_target(target_input, st.session_state.user)
                st.success("✅ Daily target set!")
                st.rerun()

    with col2:
        # Progress tracking
        st.markdown("### 📊 Progress Tracking")
        
        if current_target:
            # Progress metrics
            target = current_target
            remaining = max(0, target - today_progress)
            progress_pct = min(100, (today_progress / target) * 100)
            
            # Visual progress
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("🎯 Target", target)
            with col_b:
                st.metric("✅ Complete", today_progress)
            with col_c:
                st.metric("⏳ Remaining", remaining)
            
            # Progress bar
            st.progress(progress_pct / 100, text=f"{progress_pct:.0f}% complete")
            
            # Status messages
            if today_progress >= target:
                st.success("🎉 **Daily target achieved!** Amazing work!")
                if today_progress > target:
                    bonus = today_progress - target
                    st.balloons()
                    st.info(f"🚀 **Bonus:** +{bonus} extra session{'s' if bonus != 1 else ''}!")
            elif remaining == 1:
                st.warning("🔥 **One more to go!** You're almost there!")
            else:
                st.info(f"💪 **{remaining} sessions remaining** - you've got this!")
                
            # Motivational message
            if today_progress > 0:
                st.markdown("---")
                st.markdown("### 🧠 Freedom to Focus")
                st.info("🎯 **Work on whatever feels right!** Your goal is simply to complete your daily target of focused work.")
        else:
            st.info("👆 Set your daily target to start tracking progress")

# === DAILY GOAL COMPONENT ===
def render_daily_goal(df):
    """Render daily goal progress section"""
    if df.empty:
        active_days, today_progress, today_minutes = 0, 0, 0
        adaptive_goal, phase_name, phase_desc = 1, "🚀 Start", "Begin your journey"
    else:
        today = datetime.now(IST).date()
        df_work = df[df["pomodoro_type"] == "Work"]
        work_today = df_work[df_work["date"].dt.date == today]
        
        active_days = len(df_work.groupby(df_work["date"].dt.date).size())
        today_progress = len(work_today)
        today_minutes = work_today['duration'].sum()
        adaptive_goal, phase_name, phase_desc = get_adaptive_goal(active_days)

    return today_progress, adaptive_goal, today_minutes

# === QUICK START COMPONENT ===
def render_quick_start():
    """Render session start interface"""
    st.subheader("🚀 Quick Start")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Category management
        cat_options = st.session_state.custom_categories + ["+ Add New"]
        category_select = st.selectbox("📂 Category", cat_options, key="cat_select")

        if category_select == "+ Add New":
            new_cat = st.text_input("New category", placeholder="e.g., Marketing", key="new_cat_input")
            if new_cat and st.button("✅ Add Category", key="add_cat_btn"):
                if new_cat not in st.session_state.custom_categories:
                    st.session_state.custom_categories.append(new_cat)
                    st.session_state.category = new_cat
                    st.success("Category added!")
                    st.rerun()
            st.session_state.category = new_cat if new_cat else ""
        else:
            st.session_state.category = category_select

    with col2:
        st.session_state.task = st.text_input("🎯 Task", 
                                            placeholder="What are you working on?", 
                                            key="task_input")

    # Action buttons
    col_work, col_break = st.columns(2)

    with col_work:
        work_disabled = not st.session_state.task.strip()
        if st.button("▶️ Start Work (25min)", 
                    use_container_width=True, 
                    type="primary",
                    disabled=work_disabled):
            st.session_state.start_time = time.time()
            st.session_state.is_break = False
            st.rerun()
        
        if work_disabled:
            st.caption("⚠️ Enter a task to start working")

    with col_break:
        if st.button("☕ Break (5min)", use_container_width=True):
            st.session_state.start_time = time.time()
            st.session_state.is_break = True
            st.session_state.category = ""
            st.session_state.task = ""
            st.rerun()

# === NOTES FUNCTIONALITY ===
def add_note(content, date, user):
    """Save note to database"""
    note_id = hashlib.sha256(f"{date}_{content}_{user}".encode("utf-8")).hexdigest()
    note_doc = {
        "_id": note_id,
        "type": "Note",
        "date": date,
        "content": content,
        "user": user,
        "created_at": datetime.utcnow()
    }
    collection.update_one({"_id": note_id}, {"$set": note_doc}, upsert=True)

# === PAGE COMPONENTS ===
def render_focus_timer_page():
    """Render the main focus timer page"""
    df = get_user_data(st.session_state.user)
    
    # Check for active timer first
    if render_timer():
        return
    
    # Get today's progress
    today_progress, adaptive_goal, today_minutes = render_daily_goal(df)
    
    # Daily Target Planner (main feature)
    render_daily_target_planner(df, today_progress)
    st.divider()
    
    # Quick start section
    render_quick_start()
    
    # Today's summary
    if not df.empty:
        st.divider()
        st.subheader("📊 Today's Summary")
        
        today = datetime.now(IST).date()
        today_data = df[df["date"].dt.date == today]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Work Sessions", today_progress)
        with col2:
            st.metric("⏱️ Focus Minutes", today_minutes)
        with col3:
            breaks_today = len(today_data[today_data["pomodoro_type"] == "Break"])
            st.metric("☕ Breaks Taken", breaks_today)
        with col4:
            # Show target vs adaptive goal
            current_target = get_daily_target(st.session_state.user)
            if current_target:
                if today_progress >= current_target:
                    st.metric("✅ Target Status", "Achieved!")
                else:
                    remaining = current_target - today_progress
                    st.metric("🎯 Target Left", remaining)
            else:
                if today_progress >= adaptive_goal:
                    st.metric("✅ Goal Status", "Achieved!")
                else:
                    remaining = adaptive_goal - today_progress
                    st.metric("🎯 Sessions Left", remaining)

def render_analytics_page():
    """Render analytics dashboard"""
    st.header("📊 Analytics Dashboard")
    
    df = get_user_data(st.session_state.user)
    
    if df.empty:
        st.info("📈 Analytics will appear after your first session")
        return
    
    df_work = df[df["pomodoro_type"] == "Work"]
    today = datetime.now(IST).date()

    # Key metrics
    st.subheader("📈 Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🎯 Total Sessions", len(df_work))
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

    # Daily performance chart
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

    # Category and task breakdowns
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
        task_data = df_work.groupby('task')['duration'].sum().sort_values(ascending=False).head(8)

        if len(task_data) > 0:
            fig = px.bar(x=task_data.values, y=task_data.index, 
                       orientation='h', title="Top Tasks by Time")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    # Streak information
    st.divider()
    st.subheader("🔥 Consistency Tracking")

    daily_counts = df_work.groupby(df_work["date"].dt.date).size()
    active_days = len(daily_counts)
    min_sessions = 1 if active_days <= 12 else 2

    # Calculate current streak
    current_streak = 0
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
        # Calculate best streak
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
        recent_days = [daily_counts.get(today - timedelta(days=i), 0) for i in range(7)]
        consistency = len([d for d in recent_days if d >= min_sessions]) / 7 * 100
        st.metric("📊 Weekly Consistency", f"{consistency:.0f}%")

    # Streak explanation
    if active_days <= 12:
        st.info("💡 Building phase: 1 session per day maintains your streak")
    else:
        st.info("💡 Growth phase: 2+ sessions per day needed to maintain streak")

def render_notes_saver_page():
    """Render notes saving interface"""
    st.header("📝 Daily Notes")

    with st.form("note_form", clear_on_submit=True):
        col1, col2 = st.columns([1, 3])
        
        with col1:
            note_date = st.date_input("📅 Date", datetime.now(IST))
        with col2:
            note_content = st.text_area("✍️ Your thoughts...", 
                                      placeholder="What did you learn today?", 
                                      height=150)

        if st.form_submit_button("💾 Save Note", use_container_width=True):
            if note_content.strip():
                add_note(note_content.strip(), note_date.isoformat(), st.session_state.user)
                st.success("✅ Note saved!")
            else:
                st.warning("⚠️ Please add some content")

def render_notes_viewer_page():
    """Render notes viewing interface"""
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
    notes = list(collection.find(notes_query).sort("date", -1))

    if notes:
        for note in notes:
            with st.container():
                st.subheader(f"📅 {note['date']}")
                st.write(note['content'])
                st.divider()
    else:
        st.info("📭 No notes found in this date range")

# === MAIN APPLICATION ===
def main():
    """Main application entry point"""
    render_header()
    st.divider()
    
    # Route to appropriate page
    if st.session_state.page == "🎯 Focus Timer":
        render_focus_timer_page()
    elif st.session_state.page == "📝 Notes Saver":
        render_notes_saver_page()
    elif st.session_state.page == "📊 Analytics":
        render_analytics_page()
    elif st.session_state.page == "🗂️ Notes Viewer":
        render_notes_viewer_page()

if __name__ == "__main__":
    main()
