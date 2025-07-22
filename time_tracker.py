
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
            
            # Enhanced circular progress visualization
            st.markdown("#### 🎯 Today's Journey")
            
            # Create circular progress indicator using HTML/CSS
            circle_color = "#10b981" if today_progress >= target else "#3b82f6" if progress_pct >= 50 else "#f59e0b"
            
            circular_progress_html = f"""
            <div style="text-align: center; margin: 20px 0;">
                <div style="position: relative; display: inline-block;">
                    <svg width="150" height="150" style="transform: rotate(-90deg);">
                        <circle cx="75" cy="75" r="65" stroke="#e5e7eb" stroke-width="8" fill="none"/>
                        <circle cx="75" cy="75" r="65" stroke="{circle_color}" stroke-width="8" fill="none"
                                stroke-dasharray="408.4" stroke-dashoffset="{408.4 * (1 - progress_pct/100)}"
                                style="transition: stroke-dashoffset 0.5s ease-in-out;"/>
                    </svg>
                    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);">
                        <div style="font-size: 2rem; font-weight: bold; color: {circle_color};">{today_progress}</div>
                        <div style="font-size: 0.8rem; color: #6b7280;">of {target}</div>
                    </div>
                </div>
            </div>
            """
            st.markdown(circular_progress_html, unsafe_allow_html=True)
            
            # Enhanced metrics with color coding
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if today_progress >= target:
                    st.success("🎯 **Target**")
                    st.markdown(f"<h3 style='color: #10b981; margin: 0;'>{target}</h3>", unsafe_allow_html=True)
                else:
                    st.info("🎯 **Target**")
                    st.markdown(f"<h3 style='color: #3b82f6; margin: 0;'>{target}</h3>", unsafe_allow_html=True)
                    
            with col_b:
                if today_progress >= target:
                    st.success("✅ **Complete**")
                    st.markdown(f"<h3 style='color: #10b981; margin: 0;'>{today_progress}</h3>", unsafe_allow_html=True)
                elif today_progress > 0:
                    st.info("⚡ **Progress**")
                    st.markdown(f"<h3 style='color: #3b82f6; margin: 0;'>{today_progress}</h3>", unsafe_allow_html=True)
                else:
                    st.warning("🚀 **Start**")
                    st.markdown(f"<h3 style='color: #f59e0b; margin: 0;'>{today_progress}</h3>", unsafe_allow_html=True)
                    
            with col_c:
                if remaining == 0:
                    st.success("🏆 **Bonus Zone**")
                    bonus = today_progress - target
                    st.markdown(f"<h3 style='color: #10b981; margin: 0;'>+{bonus}</h3>", unsafe_allow_html=True)
                elif remaining == 1:
                    st.warning("🔥 **Final Push**")
                    st.markdown(f"<h3 style='color: #f59e0b; margin: 0;'>{remaining}</h3>", unsafe_allow_html=True)
                else:
                    st.info("⏳ **Remaining**")
                    st.markdown(f"<h3 style='color: #3b82f6; margin: 0;'>{remaining}</h3>", unsafe_allow_html=True)
            
            # Enhanced progress bar with percentage
            progress_text = f"🎯 {progress_pct:.0f}% Complete"
            if today_progress >= target:
                st.progress(1.0, text="🎉 Target Achieved!")
            else:
                st.progress(progress_pct / 100, text=progress_text)
            
            # Enhanced status messages with better visual hierarchy
            st.markdown("---")
            
            if today_progress >= target:
                st.success("🎉 **DAILY TARGET ACHIEVED!** 🎉")
                st.markdown("### ✨ Outstanding work today!")
                if today_progress > target:
                    bonus = today_progress - target
                    st.balloons()
                    st.markdown(f"""
                    <div style="background: linear-gradient(90deg, #10b981, #059669); color: white; padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0;">
                        <h4 style="margin: 0; color: white;">🚀 BONUS ACHIEVEMENT 🚀</h4>
                        <p style="margin: 5px 0 0 0; color: white;">+{bonus} extra session{'s' if bonus != 1 else ''}! You're on fire!</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            elif remaining == 1:
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #f59e0b, #d97706); color: white; padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0;">
                    <h4 style="margin: 0; color: white;">🔥 FINAL STRETCH! 🔥</h4>
                    <p style="margin: 5px 0 0 0; color: white;">Just one more session to hit your target!</p>
                </div>
                """, unsafe_allow_html=True)
                
            elif remaining > 1:
                motivation_messages = [
                    "You've got this! 💪",
                    "Stay focused! 🎯", 
                    "Every session counts! ⚡",
                    "Progress in action! 🚀"
                ]
                import random
                message = random.choice(motivation_messages)
                
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #3b82f6, #2563eb); color: white; padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0;">
                    <h4 style="margin: 0; color: white;">💪 KEEP GOING!</h4>
                    <p style="margin: 5px 0 0 0; color: white;">{remaining} sessions remaining - {message}</p>
                </div>
                """, unsafe_allow_html=True)
                
            else:  # remaining == 0 and today_progress == target
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #10b981, #059669); color: white; padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0;">
                    <h4 style="margin: 0; color: white;">🎯 PERFECT HIT!</h4>
                    <p style="margin: 5px 0 0 0; color: white;">Target achieved exactly! Master of focus!</p>
                </div>
                """, unsafe_allow_html=True)
                
            # Enhanced freedom message
            if today_progress > 0:
                st.markdown("---")
                st.markdown("### 🧠 Freedom to Focus")
                st.markdown(f"""
                <div style="background: #f8fafc; border-left: 4px solid #3b82f6; padding: 15px; margin: 10px 0; border-radius: 0 8px 8px 0;">
                    <p style="margin: 0; color: #475569;">
                        <strong>🎯 Work on whatever feels right!</strong><br>
                        Your goal is simply to complete <strong>{remaining} more session{'s' if remaining != 1 else ''}</strong> of focused work. 
                        Follow your energy and intuition!
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Enhanced call-to-action for setting target
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 12px; text-align: center; margin: 10px 0;">
                <h4 style="margin: 0 0 10px 0; color: white;">🎯 Ready to Focus?</h4>
                <p style="margin: 0; color: white;">Set your daily target to unlock enhanced progress tracking!</p>
            </div>
            """, unsafe_allow_html=True)

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
    
    # Enhanced Today's summary
    if not df.empty:
        st.divider()
        st.subheader("📊 Today's Summary")
        
        today = datetime.now(IST).date()
        today_data = df[df["date"].dt.date == today]
        breaks_today = len(today_data[today_data["pomodoro_type"] == "Break"])
        
        # Enhanced metrics with visual indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Work sessions with progress indicator
            current_target = get_daily_target(st.session_state.user)
            target_val = current_target if current_target else adaptive_goal
            
            if today_progress >= target_val:
                st.success("🎯 Work Sessions")
                st.markdown(f"<h2 style='color: #10b981; margin: 0;'>{today_progress}</h2>", unsafe_allow_html=True)
                st.markdown("✅ Target hit!")
            elif today_progress > 0:
                st.info("🎯 Work Sessions") 
                st.markdown(f"<h2 style='color: #3b82f6; margin: 0;'>{today_progress}</h2>", unsafe_allow_html=True)
                remaining = target_val - today_progress
                st.markdown(f"🔥 {remaining} to go!")
            else:
                st.warning("🎯 Work Sessions")
                st.markdown(f"<h2 style='color: #f59e0b; margin: 0;'>{today_progress}</h2>", unsafe_allow_html=True)
                st.markdown("🚀 Let's start!")
                
        with col2:
            # Focus minutes with time indicator
            hours = today_minutes // 60
            mins = today_minutes % 60
            
            if today_minutes >= 120:  # 2+ hours
                st.success("⏱️ Focus Time")
                if hours > 0:
                    st.markdown(f"<h2 style='color: #10b981; margin: 0;'>{hours}h {mins}m</h2>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='color: #10b981; margin: 0;'>{mins}m</h2>", unsafe_allow_html=True)
                st.markdown("🔥 Deep work!")
            elif today_minutes >= 25:
                st.info("⏱️ Focus Time")
                if hours > 0:
                    st.markdown(f"<h2 style='color: #3b82f6; margin: 0;'>{hours}h {mins}m</h2>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='color: #3b82f6; margin: 0;'>{mins}m</h2>", unsafe_allow_html=True)
                st.markdown("💪 Building up!")
            else:
                st.warning("⏱️ Focus Time")
                st.markdown(f"<h2 style='color: #f59e0b; margin: 0;'>{today_minutes}m</h2>", unsafe_allow_html=True)
                st.markdown("⚡ Just started!")
                
        with col3:
            # Break balance indicator
            work_break_ratio = breaks_today / max(1, today_progress)
            
            if 0.3 <= work_break_ratio <= 0.7:  # Good balance
                st.success("☕ Break Balance")
                st.markdown(f"<h2 style='color: #10b981; margin: 0;'>{breaks_today}</h2>", unsafe_allow_html=True)
                st.markdown("⚖️ Well balanced!")
            elif work_break_ratio > 0.7:  # Too many breaks
                st.warning("☕ Break Balance")
                st.markdown(f"<h2 style='color: #f59e0b; margin: 0;'>{breaks_today}</h2>", unsafe_allow_html=True)
                st.markdown("🎯 More focus!")
            else:  # Too few breaks
                st.info("☕ Break Balance")
                st.markdown(f"<h2 style='color: #3b82f6; margin: 0;'>{breaks_today}</h2>", unsafe_allow_html=True)
                st.markdown("🧘 Take breaks!")
                
        with col4:
            # Enhanced target status
            current_target = get_daily_target(st.session_state.user)
            
            if current_target:
                if today_progress >= current_target:
                    if today_progress > current_target:
                        st.success("🚀 Bonus Zone")
                        bonus = today_progress - current_target
                        st.markdown(f"<h2 style='color: #10b981; margin: 0;'>+{bonus}</h2>", unsafe_allow_html=True)
                        st.markdown("🌟 Exceeding!")
                    else:
                        st.success("✅ Target Hit")
                        st.markdown(f"<h2 style='color: #10b981; margin: 0;'>100%</h2>", unsafe_allow_html=True)
                        st.markdown("🎯 Perfect!")
                else:
                    remaining = current_target - today_progress
                    progress_pct = (today_progress / current_target) * 100
                    st.info("🎯 Progress")
                    st.markdown(f"<h2 style='color: #3b82f6; margin: 0;'>{progress_pct:.0f}%</h2>", unsafe_allow_html=True)
                    st.markdown(f"⏳ {remaining} left!")
            else:
                if today_progress >= adaptive_goal:
                    st.success("✅ Goal Hit")
                    st.markdown(f"<h2 style='color: #10b981; margin: 0;'>100%</h2>", unsafe_allow_html=True)
                    st.markdown("🎉 Adaptive goal!")
                else:
                    remaining = adaptive_goal - today_progress
                    progress_pct = (today_progress / adaptive_goal) * 100 if adaptive_goal > 0 else 0
                    st.info("🎯 Progress")
                    st.markdown(f"<h2 style='color: #3b82f6; margin: 0;'>{progress_pct:.0f}%</h2>", unsafe_allow_html=True)
                    st.markdown(f"⏳ {remaining} left!")

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

    # Enhanced Category and Task Analysis
    st.divider()
    st.subheader("🎯 Time Investment Analysis")
    
    # Time period filter
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    with col_filter1:
        time_filter = st.selectbox("📅 Time Period", 
                                 ["Last 7 days", "Last 30 days", "All time"], 
                                 index=1, key="time_filter_analytics")
    
    # Filter data based on selection
    if time_filter == "Last 7 days":
        cutoff_date = today - timedelta(days=7)
        filtered_work = df_work[df_work["date"].dt.date >= cutoff_date]
    elif time_filter == "Last 30 days":
        cutoff_date = today - timedelta(days=30)
        filtered_work = df_work[df_work["date"].dt.date >= cutoff_date]
    else:
        filtered_work = df_work
    
    if filtered_work.empty:
        st.info(f"📊 No data available for {time_filter.lower()}")
        return
    
    # Enhanced Category Analysis
    st.markdown("### 📂 Category Deep Dive")
    
    category_stats = filtered_work.groupby('category').agg({
        'duration': ['sum', 'count', 'mean']
    }).round(1)
    category_stats.columns = ['total_minutes', 'sessions', 'avg_session']
    category_stats = category_stats.sort_values('total_minutes', ascending=False)
    
    # Category overview metrics
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Enhanced donut chart with better styling
        if len(category_stats) > 0:
            total_time = category_stats['total_minutes'].sum()
            category_stats['percentage'] = (category_stats['total_minutes'] / total_time * 100).round(1)
            
            # Create enhanced donut chart
            fig_donut = px.pie(
                values=category_stats['total_minutes'], 
                names=category_stats.index,
                title=f"📊 Time Distribution by Category ({time_filter})",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            # Enhanced styling
            fig_donut.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>' +
                             'Time: %{value} minutes<br>' +
                             'Percentage: %{percent}<br>' +
                             '<extra></extra>',
                textfont_size=12
            )
            
            fig_donut.update_layout(
                height=400,
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
                title_x=0.5,
                font_size=12
            )
            
            # Add center text showing total time
            total_hours = total_time // 60
            total_mins = total_time % 60
            center_text = f"{total_hours}h {total_mins}m" if total_hours > 0 else f"{total_mins}m"
            
            fig_donut.add_annotation(
                text=f"<b>Total</b><br>{center_text}",
                x=0.5, y=0.5,
                font_size=16,
                showarrow=False
            )
            
            st.plotly_chart(fig_donut, use_container_width=True)
    
    with col2:
        # Category performance table
        st.markdown("#### 📈 Category Performance")
        
        # Create enhanced metrics table
        performance_data = []
        for cat in category_stats.index:
            total_mins = category_stats.loc[cat, 'total_minutes']
            sessions = int(category_stats.loc[cat, 'sessions'])
            avg_session = category_stats.loc[cat, 'avg_session']
            percentage = category_stats.loc[cat, 'percentage']
            
            # Convert to hours if > 60 minutes
            if total_mins >= 60:
                hours = int(total_mins // 60)
                mins = int(total_mins % 60)
                time_str = f"{hours}h {mins}m" if mins > 0 else f"{hours}h"
            else:
                time_str = f"{int(total_mins)}m"
            
            performance_data.append({
                'Category': cat,
                'Time': time_str,
                'Sessions': sessions,
                'Avg/Session': f"{avg_session:.0f}m",
                '%': f"{percentage:.1f}%"
            })
        
        # Display as styled dataframe
        perf_df = pd.DataFrame(performance_data)
        st.dataframe(
            perf_df, 
            use_container_width=True,
            hide_index=True,
            height=min(len(perf_df) * 35 + 38, 300)
        )
        
        # Top category highlight
        if len(category_stats) > 0:
            top_category = category_stats.index[0]
            top_percentage = category_stats.loc[top_category, 'percentage']
            
            if top_percentage > 50:
                st.success(f"🎯 **{top_category}** dominates your focus ({top_percentage:.0f}%)")
            elif top_percentage > 30:
                st.info(f"🎯 **{top_category}** is your main focus ({top_percentage:.0f}%)")
            else:
                st.warning("⚖️ Your time is well-distributed across categories")
    
    # Enhanced Task Analysis
    st.markdown("### 🎯 Task Performance Analysis")
    
    # Task stats with category context
    task_stats = filtered_work.groupby(['category', 'task']).agg({
        'duration': ['sum', 'count', 'mean']
    }).round(1)
    task_stats.columns = ['total_minutes', 'sessions', 'avg_session']
    task_stats = task_stats.reset_index().sort_values('total_minutes', ascending=False)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Enhanced horizontal bar chart for tasks
        top_tasks = task_stats.head(12)  # Show more tasks
        
        if len(top_tasks) > 0:
            # Create color mapping based on categories
            category_colors = px.colors.qualitative.Set3
            color_map = {cat: category_colors[i % len(category_colors)] 
                        for i, cat in enumerate(top_tasks['category'].unique())}
            top_tasks['color'] = top_tasks['category'].map(color_map)
            
            # Create enhanced horizontal bar chart
            fig_tasks = px.bar(
                top_tasks, 
                x='total_minutes', 
                y='task',
                color='category',
                title=f"🎯 Top Tasks by Time Investment ({time_filter})",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hover_data={
                    'sessions': True,
                    'avg_session': ':.0f',
                    'category': False
                }
            )
            
            # Enhanced styling
            fig_tasks.update_traces(
                hovertemplate='<b>%{y}</b><br>' +
                             'Category: %{color}<br>' +
                             'Time: %{x} minutes<br>' +
                             'Sessions: %{customdata[0]}<br>' +
                             'Avg/Session: %{customdata[1]:.0f}m<br>' +
                             '<extra></extra>'
            )
            
            fig_tasks.update_layout(
                height=max(400, len(top_tasks) * 30),
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title="Time (minutes)",
                yaxis_title="Tasks",
                title_x=0.5,
                showlegend=True,
                legend=dict(
                    title="Category",
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_tasks, use_container_width=True)
    
    with col2:
        # Enhanced Task insights and recommendations
        st.markdown("#### 💡 Smart Insights & Recommendations")
        
        if len(task_stats) > 0:
            # Calculate advanced insights
            total_tasks = len(task_stats)
            avg_task_time = task_stats['total_minutes'].mean()
            top_task = task_stats.iloc[0]
            total_time_invested = task_stats['total_minutes'].sum()
            
            # === TOP PERFORMER INSIGHT ===
            top_task_pct = (top_task['total_minutes'] / total_time_invested) * 100
            top_productivity_score = (top_task['sessions'] * top_task['avg_session']) / 60  # Hours of effective work
            
            # Dynamic top task analysis
            if top_task_pct > 40:
                insight_level = "🔥 DOMINANT"
                insight_color = "#dc2626"
                insight_bg = "#fef2f2"
                recommendation = "Consider balancing with other priorities"
            elif top_task_pct > 25:
                insight_level = "🎯 PRIMARY"
                insight_color = "#3b82f6"
                insight_bg = "#f0f9ff"
                recommendation = "Great focus! Maintaining momentum"
            else:
                insight_level = "⚖️ BALANCED"
                insight_color = "#059669"
                insight_bg = "#f0fdf4"
                recommendation = "Well-distributed effort across tasks"
            
            st.markdown(f"""
            <div style="background: {insight_bg}; border: 1px solid {insight_color}; border-radius: 8px; padding: 16px; margin: 8px 0;">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 1.1em; font-weight: bold; color: {insight_color};">{insight_level} FOCUS</span>
                </div>
                <div style="font-weight: bold; font-size: 1.1em; color: #1f2937; margin-bottom: 4px;">
                    {top_task['task'][:30]}{'...' if len(top_task['task']) > 30 else ''}
                </div>
                <div style="color: #6b7280; font-size: 0.9em; margin-bottom: 8px;">
                    {top_task['category']} • {int(top_task['total_minutes'])}min • {int(top_task['sessions'])} sessions
                </div>
                <div style="background: rgba(59, 130, 246, 0.1); border-radius: 4px; padding: 8px; margin-bottom: 8px;">
                    <div style="font-size: 0.85em; color: #374151;">
                        <strong>{top_task_pct:.0f}%</strong> of your total focus time
                    </div>
                </div>
                <div style="font-size: 0.8em; color: #6b7280; font-style: italic;">
                    💡 {recommendation}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # === PRODUCTIVITY PATTERNS ===
            st.markdown("##### 🔍 Productivity Patterns")
            
            # Deep work analysis
            deep_work_tasks = task_stats[task_stats['avg_session'] >= 23]  # Close to full pomodoro
            shallow_work_tasks = task_stats[task_stats['avg_session'] < 15]  # Interrupted sessions
            
            deep_work_pct = len(deep_work_tasks) / len(task_stats) * 100 if len(task_stats) > 0 else 0
            deep_work_time_pct = deep_work_tasks['total_minutes'].sum() / task_stats['total_minutes'].sum() * 100 if len(deep_work_tasks) > 0 else 0
            
            # Focus quality assessment
            if deep_work_pct >= 60:
                focus_quality = "🚀 EXCELLENT"
                focus_color = "#059669"
                focus_advice = "You excel at sustained focus!"
            elif deep_work_pct >= 40:
                focus_quality = "💪 STRONG"
                focus_color = "#3b82f6"
                focus_advice = "Good focus habits developing"
            elif deep_work_pct >= 25:
                focus_quality = "⚡ BUILDING"
                focus_color = "#f59e0b"
                focus_advice = "Room for deeper concentration"
            else:
                focus_quality = "🎯 DEVELOPING"
                focus_color = "#dc2626"
                focus_advice = "Focus on fewer interruptions"
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown(f"""
                <div style="background: #f8fafc; border: 1px solid {focus_color}; border-radius: 6px; padding: 12px; text-align: center;">
                    <div style="font-weight: bold; color: {focus_color}; font-size: 0.9em;">{focus_quality}</div>
                    <div style="font-size: 1.5em; font-weight: bold; color: #1f2937; margin: 4px 0;">{deep_work_pct:.0f}%</div>
                    <div style="font-size: 0.75em; color: #6b7280;">Deep Focus Tasks</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                # Task switching analysis
                categories_count = task_stats['category'].nunique()
                switching_score = min(100, (categories_count / max(1, len(task_stats) / 3)) * 100)
                
                if switching_score <= 50:
                    switch_quality = "🎯 FOCUSED"
                    switch_color = "#059669"
                elif switching_score <= 75:
                    switch_quality = "⚖️ BALANCED"
                    switch_color = "#3b82f6"
                else:
                    switch_quality = "🔄 DIVERSE"
                    switch_color = "#f59e0b"
                
                st.markdown(f"""
                <div style="background: #f8fafc; border: 1px solid {switch_color}; border-radius: 6px; padding: 12px; text-align: center;">
                    <div style="font-weight: bold; color: {switch_color}; font-size: 0.9em;">{switch_quality}</div>
                    <div style="font-size: 1.5em; font-weight: bold; color: #1f2937; margin: 4px 0;">{categories_count}</div>
                    <div style="font-size: 0.75em; color: #6b7280;">Categories</div>
                </div>
                """, unsafe_allow_html=True)
            
            # === ACTIONABLE RECOMMENDATIONS ===
            st.markdown("##### 🎯 Smart Recommendations")
            
            recommendations = []
            
            # Time investment recommendation
            if top_task_pct > 50:
                recommendations.append({
                    "icon": "⚖️",
                    "title": "Diversify Focus",
                    "desc": f"Consider allocating time to other important areas",
                    "priority": "medium"
                })
            
            # Deep work recommendation
            if deep_work_pct < 40:
                recommendations.append({
                    "icon": "🧠",
                    "title": "Enhance Deep Work",
                    "desc": "Try longer, uninterrupted focus sessions",
                    "priority": "high"
                })
            elif len(shallow_work_tasks) > len(task_stats) * 0.3:
                recommendations.append({
                    "icon": "🔧",
                    "title": "Reduce Interruptions",
                    "desc": f"{len(shallow_work_tasks)} tasks had short sessions",
                    "priority": "medium"
                })
            
            # Task management recommendation
            if len(task_stats) > 8:
                recommendations.append({
                    "icon": "📋",
                    "title": "Task Consolidation",
                    "desc": f"Consider grouping some of your {len(task_stats)} tasks",
                    "priority": "low"
                })
            
            # Session efficiency recommendation
            avg_efficiency = task_stats['avg_session'].mean()
            if avg_efficiency < 20:
                recommendations.append({
                    "icon": "⏰",
                    "title": "Extend Sessions",
                    "desc": f"Average {avg_efficiency:.0f}min - aim for 25min blocks",
                    "priority": "high"
                })
            
            # Display recommendations with priority styling
            for i, rec in enumerate(recommendations[:3]):  # Show top 3 recommendations
                priority_colors = {
                    "high": {"bg": "#fef2f2", "border": "#dc2626", "text": "#991b1b"},
                    "medium": {"bg": "#fffbeb", "border": "#f59e0b", "text": "#92400e"},
                    "low": {"bg": "#f0f9ff", "border": "#3b82f6", "text": "#1e40af"}
                }
                
                colors = priority_colors[rec["priority"]]
                
                st.markdown(f"""
                <div style="background: {colors['bg']}; border-left: 3px solid {colors['border']}; padding: 10px; margin: 6px 0; border-radius: 0 4px 4px 0;">
                    <div style="font-weight: bold; color: {colors['text']}; font-size: 0.9em;">
                        {rec['icon']} {rec['title']}
                    </div>
                    <div style="color: #4b5563; font-size: 0.8em; margin-top: 2px;">
                        {rec['desc']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # === WEEKLY MOMENTUM INDICATOR ===
            if time_filter != "Last 7 days":
                st.markdown("##### 🔥 Focus Momentum")
                
                # Calculate momentum based on recent activity
                cutoff_datetime = datetime.combine(today - timedelta(days=7), datetime.min.time())
                recent_7_days = task_stats[task_stats.index.isin(
                    filtered_work[filtered_work['date'] >= cutoff_datetime].groupby(['category', 'task']).first().index
                )] if len(filtered_work) > 0 else pd.DataFrame()
                
                if len(recent_7_days) > 0:
                    recent_total_time = recent_7_days['total_minutes'].sum()
                    momentum_score = min(100, (recent_total_time / max(1, total_time_invested * 0.3)) * 100)
                    
                    if momentum_score >= 80:
                        momentum_status = "🚀 ON FIRE"
                        momentum_color = "#dc2626"
                        momentum_desc = "Exceptional recent focus!"
                    elif momentum_score >= 60:
                        momentum_status = "🔥 STRONG"
                        momentum_color = "#f59e0b"
                        momentum_desc = "Great momentum building"
                    elif momentum_score >= 40:
                        momentum_status = "⚡ STEADY"
                        momentum_color = "#3b82f6"
                        momentum_desc = "Consistent progress"
                    else:
                        momentum_status = "🌱 BUILDING"
                        momentum_color = "#6b7280"
                        momentum_desc = "Time to ramp up focus"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(90deg, rgba(59, 130, 246, 0.1), rgba(99, 102, 241, 0.1)); 
                                border: 1px solid {momentum_color}; border-radius: 6px; padding: 12px; text-align: center; margin: 8px 0;">
                        <div style="font-weight: bold; color: {momentum_color}; font-size: 0.95em;">{momentum_status}</div>
                        <div style="font-size: 1.2em; font-weight: bold; color: #1f2937; margin: 4px 0;">
                            {int(recent_total_time)}min
                        </div>
                        <div style="font-size: 0.75em; color: #6b7280;">Focus time (last 7 days)</div>
                        <div style="font-size: 0.7em; color: #9ca3af; margin-top: 4px; font-style: italic;">
                            {momentum_desc}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # Enhanced empty state
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 12px; text-align: center; margin: 10px 0;">
                <h4 style="margin: 0 0 10px 0; color: white;">🎯 Ready for Insights?</h4>
                <p style="margin: 0; color: white; font-size: 0.9em;">Complete a few focus sessions to unlock smart analytics and personalized recommendations!</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recent activity pattern
        if len(filtered_work) >= 7:
            st.markdown("#### 📈 Recent Patterns")
            recent_days = filtered_work.groupby(filtered_work['date'].dt.date)['category'].nunique()
            avg_categories_per_day = recent_days.mean()
            
            if avg_categories_per_day >= 2.5:
                pattern_msg = "🌟 High variety - exploring many areas"
                pattern_color = "#10b981"
            elif avg_categories_per_day >= 1.5:
                pattern_msg = "⚖️ Moderate variety - balanced approach"
                pattern_color = "#3b82f6"
            else:
                pattern_msg = "🎯 Single focus - deep concentration"
                pattern_color = "#f59e0b"
            
            st.markdown(f"""
            <div style="background: #f8fafc; border-left: 4px solid {pattern_color}; padding: 12px; margin: 8px 0; border-radius: 4px;">
                <strong>{pattern_msg}</strong><br>
                Avg <strong>{avg_categories_per_day:.1f}</strong> categories per day
            </div>
            """, unsafe_allow_html=True)
    
    # Weekly trend analysis
    if time_filter != "Last 7 days" and len(filtered_work) > 7:
        st.markdown("### 📊 Weekly Category Trends")
        
        # Create weekly breakdown
        filtered_work['week'] = filtered_work['date'].dt.isocalendar().week
        filtered_work['year_week'] = filtered_work['date'].dt.strftime('%Y-W%U')
        
        weekly_categories = filtered_work.groupby(['year_week', 'category'])['duration'].sum().reset_index()
        
        if len(weekly_categories) > 0:
            # Create stacked bar chart for weekly trends
            fig_weekly = px.bar(
                weekly_categories, 
                x='year_week', 
                y='duration',
                color='category',
                title="📈 Weekly Time Distribution by Category",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig_weekly.update_layout(
                height=350,
                xaxis_title="Week",
                yaxis_title="Time (minutes)",
                title_x=0.5,
                legend=dict(
                    title="Category",
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            fig_weekly.update_traces(
                hovertemplate='<b>%{x}</b><br>' +
                             'Category: %{color}<br>' +
                             'Time: %{y} minutes<br>' +
                             '<extra></extra>'
            )
            
            st.plotly_chart(fig_weekly, use_container_width=True)

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
