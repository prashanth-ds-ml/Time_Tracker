
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
        "page": "🎯 Focus Timer",
        "period_targets": [{"category": "", "task": "", "daily_sessions": 1}]
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
        pages = ["🎯 Focus Timer", "📅 Period Targets", "📝 Notes Saver", "📊 Analytics", "🗂️ Notes Viewer"]
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
def generate_contribution_calendar(df):
    """Generate GitHub-style contribution calendar data"""
    today = datetime.now(IST).date()
    
    # Create 365-day grid data
    calendar_data = []
    for i in range(365):
        check_date = today - timedelta(days=364-i)
        
        if df.empty:
            day_sessions = 0
        else:
            day_data = df[(df["date"].dt.date == check_date) & (df["pomodoro_type"] == "Work")]
            day_sessions = len(day_data)
        
        # Determine intensity level (0-4 like GitHub)
        if day_sessions == 0:
            intensity = 0
        elif day_sessions == 1:
            intensity = 1
        elif day_sessions <= 3:
            intensity = 2
        elif day_sessions <= 5:
            intensity = 3
        else:
            intensity = 4
        
        # Get weekday (0 = Monday, 6 = Sunday)
        weekday = check_date.weekday()
        
        calendar_data.append({
            'date': check_date,
            'sessions': day_sessions,
            'intensity': intensity,
            'weekday': weekday,
            'week': i // 7,
            'day_name': check_date.strftime('%a'),
            'month': check_date.strftime('%b'),
            'day': check_date.day
        })
    
    return calendar_data

def render_contribution_heatmap(calendar_data):
    """Render GitHub-style contribution heatmap"""
    
    # Color mapping for intensity levels
    colors = {
        0: "#ebedf0",  # No activity
        1: "#9be9a8",  # Low activity
        2: "#40c463",  # Medium-low activity
        3: "#30a14e",  # Medium-high activity
        4: "#216e39"   # High activity
    }
    
    # Calculate total sessions for the year
    total_sessions = sum(day['sessions'] for day in calendar_data)
    active_days = len([day for day in calendar_data if day['sessions'] > 0])
    
    # Create the HTML for the contribution graph
    html_content = f"""
    <div style="background: white; padding: 20px; border-radius: 12px; border: 1px solid #e1e5e9; margin: 20px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <h3 style="margin: 0; color: #24292f;">📅 Focus Consistency (365 days)</h3>
            <div style="font-size: 14px; color: #656d76;">
                <strong>{total_sessions}</strong> sessions in the last year • <strong>{active_days}</strong> active days
            </div>
        </div>
        
        <div style="display: flex; align-items: center; gap: 12px;">
            <!-- Month labels -->
            <div style="width: 100%; overflow-x: auto;">
                <div style="display: flex; margin-bottom: 8px; gap: 11px; font-size: 11px; color: #656d76;">
    """
    
    # Add month labels
    months_shown = set()
    for week in range(0, 53):
        week_start = min(week * 7, len(calendar_data) - 1)
        if week_start < len(calendar_data):
            month = calendar_data[week_start]['month']
            if month not in months_shown:
                html_content += f"<div style='width: 12px; text-align: center;'>{month}</div>"
                months_shown.add(month)
            else:
                html_content += "<div style='width: 12px;'></div>"
    
    html_content += """
                </div>
                
                <div style="display: flex; gap: 3px;">
    """
    
    # Create the grid (53 weeks x 7 days)
    for week in range(53):
        html_content += "<div style='display: flex; flex-direction: column; gap: 3px;'>"
        
        for day in range(7):
            day_index = week * 7 + day
            if day_index < len(calendar_data):
                day_data = calendar_data[day_index]
                color = colors[day_data['intensity']]
                
                # Create tooltip text
                sessions_text = f"{day_data['sessions']} session{'s' if day_data['sessions'] != 1 else ''}"
                date_str = day_data['date'].strftime('%B %d, %Y')
                
                html_content += f"""
                <div style="width: 12px; height: 12px; background-color: {color}; border-radius: 2px; 
                           border: 1px solid rgba(1,4,9,0.1);"
                     title="{sessions_text} on {date_str}">
                </div>
                """
            else:
                html_content += "<div style='width: 12px; height: 12px;'></div>"
        
        html_content += "</div>"
    
    html_content += """
                </div>
            </div>
        </div>
        
        <!-- Legend -->
        <div style="display: flex; align-items: center; justify-content: flex-end; gap: 8px; margin-top: 16px; font-size: 12px; color: #656d76;">
            <span>Less</span>
    """
    
    # Add legend squares
    for i in range(5):
        color = colors[i]
        html_content += f'<div style="width: 10px; height: 10px; background-color: {color}; border-radius: 2px; border: 1px solid rgba(1,4,9,0.1);"></div>'
    
    html_content += """
            <span>More</span>
        </div>
    </div>
    """
    
    return html_content

def render_dashboard_metrics(df, today_progress, today_minutes):
    """Render key dashboard metrics"""
    today = datetime.now(IST).date()
    
    if df.empty:
        metrics = {
            'total_sessions': 0,
            'total_hours': 0,
            'current_streak': 0,
            'best_streak': 0,
            'weekly_avg': 0,
            'monthly_total': 0
        }
    else:
        df_work = df[df["pomodoro_type"] == "Work"]
        
        # Calculate metrics
        total_sessions = len(df_work)
        total_hours = df_work['duration'].sum() // 60
        
        # Streak calculations
        daily_counts = df_work.groupby(df_work["date"].dt.date).size()
        active_days = len(daily_counts)
        min_sessions = 1 if active_days <= 12 else 2
        
        # Current streak
        current_streak = 0
        for i in range(365):
            check_date = today - timedelta(days=i)
            day_count = daily_counts.get(check_date, 0)
            if day_count >= min_sessions:
                current_streak += 1
            else:
                break
        
        # Best streak
        best_streak = 0
        temp_streak = 0
        for i in range(365):
            check_date = today - timedelta(days=364-i)
            day_count = daily_counts.get(check_date, 0)
            if day_count >= min_sessions:
                temp_streak += 1
                best_streak = max(best_streak, temp_streak)
            else:
                temp_streak = 0
        
        # Weekly average
        last_4_weeks = df_work[df_work["date"] >= (today - timedelta(days=28))]
        weekly_avg = len(last_4_weeks) / 4
        
        # Monthly total
        this_month = df_work[df_work["date"].dt.month == today.month]
        monthly_total = len(this_month)
        
        metrics = {
            'total_sessions': total_sessions,
            'total_hours': total_hours,
            'current_streak': current_streak,
            'best_streak': best_streak,
            'weekly_avg': weekly_avg,
            'monthly_total': monthly_total
        }
    
    # Render metrics cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        streak_color = "#dc2626" if metrics['current_streak'] >= 7 else "#f59e0b" if metrics['current_streak'] >= 3 else "#3b82f6"
        st.markdown(f"""
        <div style="background: white; border: 1px solid {streak_color}; border-radius: 8px; padding: 16px; text-align: center;">
            <div style="font-size: 2em; color: {streak_color};">🔥</div>
            <div style="font-size: 1.5em; font-weight: bold; color: #1f2937; margin: 8px 0;">{metrics['current_streak']}</div>
            <div style="font-size: 0.85em; color: #6b7280;">Current Streak</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: white; border: 1px solid #10b981; border-radius: 8px; padding: 16px; text-align: center;">
            <div style="font-size: 2em; color: #10b981;">🎯</div>
            <div style="font-size: 1.5em; font-weight: bold; color: #1f2937; margin: 8px 0;">{metrics['total_sessions']}</div>
            <div style="font-size: 0.85em; color: #6b7280;">Total Sessions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: white; border: 1px solid #3b82f6; border-radius: 8px; padding: 16px; text-align: center;">
            <div style="font-size: 2em; color: #3b82f6;">⏰</div>
            <div style="font-size: 1.5em; font-weight: bold; color: #1f2937; margin: 8px 0;">{metrics['total_hours']}h</div>
            <div style="font-size: 0.85em; color: #6b7280;">Total Focus</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style="background: white; border: 1px solid #8b5cf6; border-radius: 8px; padding: 16px; text-align: center;">
            <div style="font-size: 2em; color: #8b5cf6;">🏆</div>
            <div style="font-size: 1.5em; font-weight: bold; color: #1f2937; margin: 8px 0;">{metrics['best_streak']}</div>
            <div style="font-size: 0.85em; color: #6b7280;">Best Streak</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div style="background: white; border: 1px solid #f59e0b; border-radius: 8px; padding: 16px; text-align: center;">
            <div style="font-size: 2em; color: #f59e0b;">📊</div>
            <div style="font-size: 1.5em; font-weight: bold; color: #1f2937; margin: 8px 0;">{metrics['weekly_avg']:.1f}</div>
            <div style="font-size: 0.85em; color: #6b7280;">Weekly Avg</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown(f"""
        <div style="background: white; border: 1px solid #06b6d4; border-radius: 8px; padding: 16px; text-align: center;">
            <div style="font-size: 2em; color: #06b6d4;">📅</div>
            <div style="font-size: 1.5em; font-weight: bold; color: #1f2937; margin: 8px 0;">{metrics['monthly_total']}</div>
            <div style="font-size: 0.85em; color: #6b7280;">This Month</div>
        </div>
        """, unsafe_allow_html=True)

def render_active_targets_summary():
    """Render summary of active period targets"""
    active_targets = get_active_period_targets(st.session_state.user)
    
    if not active_targets:
        return
    
    st.markdown("### 🎯 Active Period Targets")
    
    for plan in active_targets[:2]:  # Show max 2 active plans
        progress, overall_pct, days_elapsed = get_period_target_progress(plan, st.session_state.user)
        
        # Compact progress display
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            progress_color = "#10b981" if overall_pct >= 80 else "#3b82f6" if overall_pct >= 60 else "#f59e0b"
            
            st.markdown(f"""
            <div style="background: white; border-left: 4px solid {progress_color}; padding: 12px; border-radius: 0 6px 6px 0; border: 1px solid #e5e7eb;">
                <div style="font-weight: bold; color: #1f2937; margin-bottom: 4px;">
                    📋 {plan['plan_name']}
                </div>
                <div style="font-size: 0.9em; color: #6b7280; margin-bottom: 8px;">
                    📅 {plan['start_date']} → {plan['end_date']} • Day {days_elapsed}
                </div>
                <div style="background: #f3f4f6; border-radius: 10px; height: 8px; overflow: hidden;">
                    <div style="background: {progress_color}; height: 100%; width: {min(100, overall_pct):.1f}%; transition: width 0.3s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Progress", f"{overall_pct:.0f}%")
        
        with col3:
            total_targets = len(plan['targets'])
            on_track = len([p for p in progress.values() if p['percentage'] >= 80])
            st.metric("On Track", f"{on_track}/{total_targets}")
    
    # Quick link to targets page
    if st.button("📊 View All Targets", key="view_targets_btn"):
        st.session_state.page = "📅 Period Targets"
        st.rerun()

def render_focus_timer_page():
    """Render the enhanced main dashboard page"""
    df = get_user_data(st.session_state.user)
    
    # Check for active timer first
    if render_timer():
        return
    
    # Get today's progress
    today_progress, adaptive_goal, today_minutes = render_daily_goal(df)
    
    # === HERO SECTION ===
    st.markdown("## 🚀 Focus Dashboard")
    
    # Generate and display contribution calendar
    calendar_data = generate_contribution_calendar(df)
    contribution_html = render_contribution_heatmap(calendar_data)
    st.markdown(contribution_html, unsafe_allow_html=True)
    
    # === KEY METRICS ===
    st.markdown("### 📊 Your Focus Journey")
    render_dashboard_metrics(df, today_progress, today_minutes)
    
    st.divider()
    
    # === TODAY'S PROGRESS & QUICK START ===
    col_today, col_start = st.columns([1, 1])
    
    with col_today:
        # Daily Target Planner (compact version)
        render_daily_target_planner(df, today_progress)
    
    with col_start:
        # Quick start section
        render_quick_start()
    
    st.divider()
    
    # === ACTIVE TARGETS SUMMARY ===
    render_active_targets_summary()
    
    # === RECENT ACTIVITY ===
    if not df.empty:
        st.divider()
        st.markdown("### 📈 Recent Activity")
        
        # Last 7 days summary
        today = datetime.now(IST).date()
        last_7_days = []
        
        for i in range(7):
            day_date = today - timedelta(days=6-i)
            day_data = df[(df["date"].dt.date == day_date) & (df["pomodoro_type"] == "Work")]
            sessions = len(day_data)
            minutes = day_data['duration'].sum() if not day_data.empty else 0
            
            last_7_days.append({
                'date': day_date,
                'day_name': day_date.strftime('%a'),
                'sessions': sessions,
                'minutes': minutes,
                'is_today': day_date == today
            })
        
        # Display 7-day chart
        cols = st.columns(7)
        for i, day in enumerate(last_7_days):
            with cols[i]:
                # Determine intensity color
                if day['sessions'] == 0:
                    color = "#ebedf0"
                    text_color = "#6b7280"
                elif day['sessions'] <= 2:
                    color = "#9be9a8" 
                    text_color = "#1f2937"
                elif day['sessions'] <= 4:
                    color = "#40c463"
                    text_color = "#1f2937"
                else:
                    color = "#216e39"
                    text_color = "white"
                
                border_style = "border: 2px solid #3b82f6;" if day['is_today'] else "border: 1px solid #e5e7eb;"
                
                st.markdown(f"""
                <div style="background: {color}; {border_style} border-radius: 8px; padding: 12px; text-align: center; margin: 2px;">
                    <div style="font-weight: bold; color: {text_color}; font-size: 0.9em;">{day['day_name']}</div>
                    <div style="font-size: 1.2em; font-weight: bold; color: {text_color}; margin: 4px 0;">{day['sessions']}</div>
                    <div style="font-size: 0.75em; color: {text_color};">{day['minutes']}m</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Quick navigation to other features
        st.divider()
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 View Analytics", key="nav_analytics", use_container_width=True):
                st.session_state.page = "📊 Analytics"
                st.rerun()
        
        with col2:
            if st.button("🎯 Set Targets", key="nav_targets", use_container_width=True):
                st.session_state.page = "📅 Period Targets"
                st.rerun()
        
        with col3:
            if st.button("📝 Save Notes", key="nav_notes", use_container_width=True):
                st.session_state.page = "📝 Notes Saver"
                st.rerun()
        
        with col4:
            if st.button("🗂️ View Notes", key="nav_notes_view", use_container_width=True):
                st.session_state.page = "🗂️ Notes Viewer"
                st.rerun()
    
    else:
        # Welcome message for new users
        st.divider()
        st.markdown("""
        ### 🌟 Welcome to Your Focus Journey!
        
        Start your first Pomodoro session to begin tracking your consistency and building productive habits.
        
        **🎯 Quick Tips:**
        - Set a daily target to stay motivated
        - Take regular breaks to maintain focus
        - Track your progress with our GitHub-style contribution graph
        - Set weekly or monthly targets for bigger goals
        """)
        
        # Motivational call-to-action
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 12px; text-align: center; margin: 20px 0;">
            <h3 style="margin: 0 0 10px 0; color: white;">🚀 Ready to Start?</h3>
            <p style="margin: 0; color: white;">Your focus journey begins with a single session. What will you work on first?</p>
        </div>
        """, unsafe_allow_html=True)

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
        
        # Enhanced Recent Patterns Analysis
        if len(filtered_work) >= 7:
            st.markdown("#### 📈 Recent Patterns & Trends")
            
            # Calculate pattern metrics
            recent_days = filtered_work.groupby(filtered_work['date'].dt.date)['category'].nunique()
            avg_categories_per_day = recent_days.mean()
            
            # Time allocation patterns
            recent_category_time = filtered_work.groupby('category')['duration'].sum()
            recent_total_time = recent_category_time.sum()
            
            # Compare with earlier period if enough data
            if time_filter == "Last 30 days" and len(filtered_work) >= 14:
                # Split into recent vs earlier halves
                mid_date = today - timedelta(days=15)
                earlier_work = filtered_work[filtered_work['date'].dt.date < mid_date]
                recent_work = filtered_work[filtered_work['date'].dt.date >= mid_date]
                
                if len(earlier_work) > 0 and len(recent_work) > 0:
                    # Calculate percentage changes
                    earlier_cat_time = earlier_work.groupby('category')['duration'].sum()
                    recent_cat_time = recent_work.groupby('category')['duration'].sum()
                    
                    earlier_total = earlier_cat_time.sum()
                    recent_total = recent_cat_time.sum()
                    
                    # Pattern insights with percentage changes
                    pattern_changes = []
                    all_categories = set(earlier_cat_time.index) | set(recent_cat_time.index)
                    
                    for cat in all_categories:
                        earlier_pct = (earlier_cat_time.get(cat, 0) / earlier_total * 100) if earlier_total > 0 else 0
                        recent_pct = (recent_cat_time.get(cat, 0) / recent_total * 100) if recent_total > 0 else 0
                        change_pct = recent_pct - earlier_pct
                        
                        if abs(change_pct) >= 5:  # Significant change threshold
                            pattern_changes.append({
                                'category': cat,
                                'change': change_pct,
                                'recent_pct': recent_pct,
                                'earlier_pct': earlier_pct
                            })
                    
                    # Sort by absolute change magnitude
                    pattern_changes.sort(key=lambda x: abs(x['change']), reverse=True)
                    
                    # Display pattern changes
                    if pattern_changes:
                        st.markdown("##### 📊 Time Allocation Shifts (Last 15 vs Previous 15 days)")
                        
                        for i, change in enumerate(pattern_changes[:3]):  # Show top 3 changes
                            cat = change['category']
                            change_val = change['change']
                            recent_pct = change['recent_pct']
                            earlier_pct = change['earlier_pct']
                            
                            if change_val > 0:
                                trend_icon = "📈"
                                trend_color = "#059669"
                                trend_desc = "INCREASING"
                                change_text = f"+{change_val:.1f}%"
                            else:
                                trend_icon = "📉"
                                trend_color = "#dc2626"
                                trend_desc = "DECREASING"
                                change_text = f"{change_val:.1f}%"
                            
                            st.markdown(f"""
                            <div style="background: #f8fafc; border: 1px solid {trend_color}; border-radius: 6px; padding: 12px; margin: 6px 0;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div style="font-weight: bold; color: #1f2937; font-size: 0.95em;">
                                        {trend_icon} {cat}
                                    </div>
                                    <div style="font-weight: bold; color: {trend_color}; font-size: 0.9em;">
                                        {trend_desc}
                                    </div>
                                </div>
                                <div style="margin: 8px 0;">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                        <span style="font-size: 0.85em; color: #6b7280;">Previous 15 days:</span>
                                        <span style="font-weight: bold; color: #374151;">{earlier_pct:.1f}%</span>
                                    </div>
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                        <span style="font-size: 0.85em; color: #6b7280;">Recent 15 days:</span>
                                        <span style="font-weight: bold; color: #374151;">{recent_pct:.1f}%</span>
                                    </div>
                                    <div style="background: {trend_color}; color: white; text-align: center; padding: 4px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold;">
                                        {change_text} change in time allocation
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Overall pattern assessment
                    total_change = sum(abs(c['change']) for c in pattern_changes)
                    if total_change >= 20:
                        pattern_stability = "🌪️ DYNAMIC"
                        stability_color = "#dc2626"
                        stability_desc = "Significant shifts in focus areas"
                    elif total_change >= 10:
                        pattern_stability = "⚡ EVOLVING"
                        stability_color = "#f59e0b"
                        stability_desc = "Moderate changes in priorities"
                    else:
                        pattern_stability = "🎯 STABLE"
                        stability_color = "#059669"
                        stability_desc = "Consistent focus patterns"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(90deg, rgba(59, 130, 246, 0.1), rgba(99, 102, 241, 0.1)); 
                                border: 1px solid {stability_color}; border-radius: 6px; padding: 12px; text-align: center; margin: 12px 0;">
                        <div style="font-weight: bold; color: {stability_color}; font-size: 0.95em;">
                            PATTERN STATUS: {pattern_stability}
                        </div>
                        <div style="font-size: 0.8em; color: #6b7280; margin-top: 4px;">
                            {stability_desc}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Daily focus diversity analysis
            st.markdown("##### 🎯 Daily Focus Diversity")
            
            diversity_scores = []
            for i in range(min(14, len(filtered_work.groupby(filtered_work['date'].dt.date)))):
                check_date = today - timedelta(days=i)
                day_data = filtered_work[filtered_work['date'].dt.date == check_date]
                if len(day_data) > 0:
                    categories_count = day_data['category'].nunique()
                    sessions_count = len(day_data)
                    diversity_score = min(100, (categories_count / max(1, sessions_count)) * 100)
                    diversity_scores.append({
                        'date': check_date.strftime('%m/%d'),
                        'diversity': diversity_score,
                        'categories': categories_count,
                        'sessions': sessions_count
                    })
            
            if diversity_scores:
                # Recent diversity trend
                recent_avg_diversity = sum(d['diversity'] for d in diversity_scores[:7]) / min(7, len(diversity_scores))
                earlier_avg_diversity = sum(d['diversity'] for d in diversity_scores[7:]) / max(1, len(diversity_scores[7:]))
                
                diversity_trend = recent_avg_diversity - earlier_avg_diversity
                
                col_div1, col_div2, col_div3 = st.columns(3)
                
                with col_div1:
                    if recent_avg_diversity >= 60:
                        div_status = "🌟 HIGH"
                        div_color = "#059669"
                    elif recent_avg_diversity >= 30:
                        div_status = "⚖️ MODERATE"
                        div_color = "#3b82f6"
                    else:
                        div_status = "🎯 FOCUSED"
                        div_color = "#f59e0b"
                    
                    st.markdown(f"""
                    <div style="background: #f8fafc; border: 1px solid {div_color}; border-radius: 6px; padding: 12px; text-align: center;">
                        <div style="font-weight: bold; color: {div_color}; font-size: 0.9em;">{div_status}</div>
                        <div style="font-size: 1.5em; font-weight: bold; color: #1f2937; margin: 4px 0;">{recent_avg_diversity:.0f}%</div>
                        <div style="font-size: 0.75em; color: #6b7280;">Focus Diversity</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_div2:
                    if abs(diversity_trend) < 5:
                        trend_status = "➡️ STABLE"
                        trend_color = "#6b7280"
                    elif diversity_trend > 0:
                        trend_status = "📈 EXPANDING"
                        trend_color = "#059669"
                    else:
                        trend_status = "📉 NARROWING"
                        trend_color = "#dc2626"
                    
                    st.markdown(f"""
                    <div style="background: #f8fafc; border: 1px solid {trend_color}; border-radius: 6px; padding: 12px; text-align: center;">
                        <div style="font-weight: bold; color: {trend_color}; font-size: 0.9em;">{trend_status}</div>
                        <div style="font-size: 1.5em; font-weight: bold; color: #1f2937; margin: 4px 0;">
                            {diversity_trend:+.0f}%
                        </div>
                        <div style="font-size: 0.75em; color: #6b7280;">Trend (7d)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_div3:
                    avg_cats_recent = sum(d['categories'] for d in diversity_scores[:7]) / min(7, len(diversity_scores))
                    
                    if avg_cats_recent >= 3:
                        cats_status = "🌈 VARIED"
                        cats_color = "#8b5cf6"
                    elif avg_cats_recent >= 2:
                        cats_status = "⚖️ BALANCED"
                        cats_color = "#3b82f6"
                    else:
                        cats_status = "🎯 SINGULAR"
                        cats_color = "#f59e0b"
                    
                    st.markdown(f"""
                    <div style="background: #f8fafc; border: 1px solid {cats_color}; border-radius: 6px; padding: 12px; text-align: center;">
                        <div style="font-weight: bold; color: {cats_color}; font-size: 0.9em;">{cats_status}</div>
                        <div style="font-size: 1.5em; font-weight: bold; color: #1f2937; margin: 4px 0;">{avg_cats_recent:.1f}</div>
                        <div style="font-size: 0.75em; color: #6b7280;">Avg Categories/Day</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Session timing patterns
            st.markdown("##### ⏰ Session Timing Patterns")
            
            # Add hour analysis
            filtered_work['hour'] = pd.to_datetime(filtered_work['time'], format='%I:%M %p', errors='coerce').dt.hour
            hourly_sessions = filtered_work.groupby('hour').size()
            
            if len(hourly_sessions) > 0:
                peak_hours = hourly_sessions.nlargest(3)
                
                # Convert hours to readable format
                peak_times = []
                for hour, count in peak_hours.items():
                    if pd.notna(hour):
                        time_str = f"{int(hour):02d}:00"
                        if hour < 12:
                            time_str += " AM"
                        elif hour == 12:
                            time_str += " PM"
                        else:
                            time_str = f"{int(hour-12):02d}:00 PM"
                        peak_times.append((time_str, count))
                
                if peak_times:
                    # Determine productivity pattern
                    if any(hour < 12 for hour, _ in peak_hours.items() if pd.notna(hour)):
                        if any(hour >= 12 for hour, _ in peak_hours.items() if pd.notna(hour)):
                            timing_pattern = "🌅🌆 ALL-DAY WARRIOR"
                            timing_color = "#8b5cf6"
                            timing_desc = "Productive throughout the day"
                        else:
                            timing_pattern = "🌅 EARLY BIRD"
                            timing_color = "#f59e0b"
                            timing_desc = "Morning productivity focus"
                    else:
                        timing_pattern = "🌙 NIGHT OWL"
                        timing_color = "#3b82f6"
                        timing_desc = "Afternoon/evening focus"
                    
                    st.markdown(f"""
                    <div style="background: #f8fafc; border-left: 4px solid {timing_color}; padding: 12px; margin: 8px 0; border-radius: 0 6px 6px 0;">
                        <div style="font-weight: bold; color: {timing_color}; margin-bottom: 8px;">
                            {timing_pattern}
                        </div>
                        <div style="font-size: 0.9em; color: #374151; margin-bottom: 8px;">
                            {timing_desc}
                        </div>
                        <div style="font-size: 0.85em; color: #6b7280;">
                            <strong>Peak focus times:</strong> {', '.join([f"{time} ({count} sessions)" for time, count in peak_times[:2]])}
                        </div>
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

def save_period_target(plan_name, start_date, end_date, targets, user):
    """Save weekly/21-day target plan to database"""
    plan_id = hashlib.sha256(f"{plan_name}_{start_date}_{user}".encode("utf-8")).hexdigest()
    plan_doc = {
        "_id": plan_id,
        "type": "PeriodTarget",
        "plan_name": plan_name,
        "start_date": start_date,
        "end_date": end_date,
        "targets": targets,  # List of {category, task, daily_sessions}
        "user": user,
        "created_at": datetime.utcnow()
    }
    collection.update_one({"_id": plan_id}, {"$set": plan_doc}, upsert=True)

def get_active_period_targets(user):
    """Get active period targets for user"""
    today = datetime.now(IST).date().isoformat()
    targets = list(collection.find({
        "type": "PeriodTarget",
        "user": user,
        "start_date": {"$lte": today},
        "end_date": {"$gte": today}
    }).sort("created_at", -1))
    return targets

def get_period_target_progress(target_plan, user):
    """Calculate progress for a period target plan"""
    start_date = datetime.fromisoformat(target_plan["start_date"]).date()
    end_date = datetime.fromisoformat(target_plan["end_date"]).date()
    today = datetime.now(IST).date()
    
    # Get all work sessions in the period
    work_data = list(collection.find({
        "type": "Pomodoro",
        "pomodoro_type": "Work",
        "user": user,
        "date": {
            "$gte": start_date.isoformat(),
            "$lte": min(today, end_date).isoformat()
        }
    }))
    
    if not work_data:
        return {}, 0, 0
    
    df_work = pd.DataFrame(work_data)
    df_work["date"] = pd.to_datetime(df_work["date"]).dt.date
    
    # Calculate progress for each target
    progress = {}
    total_expected = 0
    total_completed = 0
    
    days_elapsed = (min(today, end_date) - start_date).days + 1
    
    for target in target_plan["targets"]:
        category = target["category"]
        task = target["task"]
        daily_sessions = target["daily_sessions"]
        
        # Filter sessions for this specific category-task combination
        target_sessions = df_work[
            (df_work["category"] == category) & 
            (df_work["task"] == task)
        ]
        
        completed_sessions = len(target_sessions)
        expected_sessions = daily_sessions * days_elapsed
        
        progress[f"{category}:{task}"] = {
            "category": category,
            "task": task,
            "daily_target": daily_sessions,
            "expected": expected_sessions,
            "completed": completed_sessions,
            "percentage": (completed_sessions / max(1, expected_sessions)) * 100,
            "remaining_daily": max(0, daily_sessions - len(target_sessions[target_sessions["date"] == today]))
        }
        
        total_expected += expected_sessions
        total_completed += completed_sessions
    
    overall_percentage = (total_completed / max(1, total_expected)) * 100
    
    return progress, overall_percentage, days_elapsed

def render_period_targets_page():
    """Render Weekly/21-Day Target Planning interface"""
    st.header("🎯 Weekly/21-Day Target Planning")
    
    # Initialize period plan state
    if 'period_plan' not in st.session_state:
        st.session_state.period_plan = {
            "name": "",
            "start_date": datetime.now(IST).date(),
            "duration_days": 7,
            "targets": [],
            "editing_target": None
        }
    
    if 'temp_target' not in st.session_state:
        st.session_state.temp_target = {
            "category": "",
            "task": "",
            "daily_sessions": 1
        }
    
    # Get active targets
    active_targets = get_active_period_targets(st.session_state.user)
    
    # === ACTIVE PLANS SECTION ===
    if active_targets:
        st.markdown("### 📊 Active Plans")
        
        for plan in active_targets:
            with st.expander(f"📋 {plan['plan_name']} ({plan['start_date']} to {plan['end_date']})", expanded=True):
                progress, overall_pct, days_elapsed = get_period_target_progress(plan, st.session_state.user)
                
                # Overall progress
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**Overall Progress: {overall_pct:.1f}%**")
                    st.progress(min(1.0, overall_pct / 100))
                
                with col2:
                    st.metric("📅 Day", f"{days_elapsed}")
                
                with col3:
                    total_days = (datetime.fromisoformat(plan['end_date']).date() - 
                                datetime.fromisoformat(plan['start_date']).date()).days + 1
                    st.metric("📊 Duration", f"{total_days} days")
                
                # Individual target progress
                st.markdown("#### 🎯 Target Progress")
                
                for target_key, prog in progress.items():
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        # Color-coded progress indicator
                        if prog["percentage"] >= 100:
                            status_color = "#10b981"
                            status_icon = "✅"
                        elif prog["percentage"] >= 80:
                            status_color = "#3b82f6"
                            status_icon = "🔵"
                        elif prog["percentage"] >= 60:
                            status_color = "#f59e0b"
                            status_icon = "🟡"
                        else:
                            status_color = "#dc2626"
                            status_icon = "🔴"
                        
                        st.markdown(f"""
                        <div style="background: #f8fafc; border-left: 4px solid {status_color}; padding: 12px; margin: 4px 0; border-radius: 0 6px 6px 0;">
                            <div style="font-weight: bold; color: #1f2937;">
                                {status_icon} {prog['task']}
                            </div>
                            <div style="font-size: 0.85em; color: #6b7280;">
                                📂 {prog['category']} • 🎯 {prog['daily_target']} sessions/day
                            </div>
                            <div style="margin-top: 6px;">
                                <div style="background: #e5e7eb; border-radius: 10px; height: 8px; overflow: hidden;">
                                    <div style="background: {status_color}; height: 100%; width: {min(100, prog['percentage']):.1f}%; transition: width 0.3s ease;"></div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if prog["percentage"] >= 100:
                            st.success(f"**{prog['percentage']:.0f}%**")
                        elif prog["percentage"] >= 80:
                            st.info(f"**{prog['percentage']:.0f}%**")
                        elif prog["percentage"] >= 60:
                            st.warning(f"**{prog['percentage']:.0f}%**")
                        else:
                            st.error(f"**{prog['percentage']:.0f}%**")
                    
                    with col3:
                        st.markdown(f"**{prog['completed']}/{prog['expected']}**")
                        st.caption("Completed")
                    
                    with col4:
                        if prog["remaining_daily"] > 0:
                            st.warning(f"**{prog['remaining_daily']}**")
                            st.caption("Today remaining")
                        else:
                            st.success("**✓**")
                            st.caption("Today done")
        
        st.divider()
    
    # === PLAN CREATION SECTION ===
    st.markdown("### ➕ Create New Plan")
    
    # === PLAN METADATA ===
    st.markdown("#### 📋 Plan Information")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        plan_name = st.text_input(
            "Plan Name", 
            value=st.session_state.period_plan["name"],
            placeholder="e.g., UGC NET Preparation Week 1",
            key="plan_name_input"
        )
        if plan_name != st.session_state.period_plan["name"]:
            st.session_state.period_plan["name"] = plan_name
    
    with col2:
        period_options = [
            ("1 Week", 7),
            ("3 Weeks", 21),
            ("1 Month", 30),
            ("Custom", 0)
        ]
        
        period_labels = [opt[0] for opt in period_options]
        current_duration = st.session_state.period_plan["duration_days"]
        
        # Find current selection
        current_idx = 0
        for i, (label, days) in enumerate(period_options):
            if days == current_duration:
                current_idx = i
                break
            elif label == "Custom":
                current_idx = i
        
        selected_period = st.selectbox(
            "Duration",
            period_labels,
            index=current_idx,
            key="period_duration_select"
        )
        
        selected_days = next(days for label, days in period_options if label == selected_period)
        
        if selected_period == "Custom":
            custom_days = st.number_input(
                "Custom Days", 
                min_value=1, 
                max_value=90, 
                value=current_duration if current_duration not in [7, 21, 30] else 7,
                key="custom_days_input"
            )
            st.session_state.period_plan["duration_days"] = custom_days
        else:
            st.session_state.period_plan["duration_days"] = selected_days
    
    # Date selection
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "🚀 Start Date", 
            value=st.session_state.period_plan["start_date"],
            key="plan_start_date"
        )
        st.session_state.period_plan["start_date"] = start_date
    
    with col2:
        end_date = start_date + timedelta(days=st.session_state.period_plan["duration_days"] - 1)
        st.date_input("🏁 End Date", value=end_date, disabled=True, key="plan_end_date_display")
    
    st.divider()
    
    # === TARGET MANAGEMENT ===
    st.markdown("#### 🎯 Daily Targets")
    
    # Current targets display
    if st.session_state.period_plan["targets"]:
        st.markdown("**Current Targets:**")
        
        for i, target in enumerate(st.session_state.period_plan["targets"]):
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            
            with col1:
                st.markdown(f"📂 **{target['category']}**")
            
            with col2:
                st.markdown(f"🎯 {target['task']}")
            
            with col3:
                st.markdown(f"**{target['daily_sessions']}** sessions/day")
            
            with col4:
                if st.button("🗑️", key=f"remove_target_{i}", help="Remove target"):
                    st.session_state.period_plan["targets"].pop(i)
                    st.rerun()
        
        st.divider()
    
    # === ADD NEW TARGET INTERFACE ===
    st.markdown("**Add New Target:**")
    
    col1, col2, col3, col4 = st.columns([2, 3, 1, 1])
    
    with col1:
        # Category selection with new category option
        available_categories = st.session_state.custom_categories.copy()
        
        category_options = available_categories + ["+ Add New Category"]
        
        selected_category = st.selectbox(
            "Category",
            category_options,
            key="new_target_category",
            help="Select existing category or add a new one"
        )
        
        if selected_category == "+ Add New Category":
            new_category = st.text_input(
                "New Category Name",
                placeholder="Enter category name",
                key="new_category_input"
            )
            
            if new_category:
                if st.button("➕ Add Category", key="add_category_btn", type="secondary"):
                    if new_category.strip() and new_category.strip() not in st.session_state.custom_categories:
                        st.session_state.custom_categories.append(new_category.strip())
                        st.session_state.temp_target["category"] = new_category.strip()
                        st.success(f"✅ Category '{new_category.strip()}' added!")
                        st.rerun()
                    else:
                        st.warning("⚠️ Category already exists or is empty")
                
                st.session_state.temp_target["category"] = new_category.strip() if new_category.strip() else ""
            else:
                st.session_state.temp_target["category"] = ""
        else:
            st.session_state.temp_target["category"] = selected_category
    
    with col2:
        task_input = st.text_input(
            "Task",
            value=st.session_state.temp_target["task"],
            placeholder="e.g., UGC NET Paper 1, SQL Projects",
            key="new_target_task"
        )
        st.session_state.temp_target["task"] = task_input
    
    with col3:
        sessions_input = st.number_input(
            "Sessions/day",
            min_value=1,
            max_value=10,
            value=st.session_state.temp_target["daily_sessions"],
            key="new_target_sessions"
        )
        st.session_state.temp_target["daily_sessions"] = sessions_input
    
    with col4:
        # Add target button
        can_add_target = (
            st.session_state.temp_target["category"] and 
            st.session_state.temp_target["category"] != "+ Add New Category" and
            st.session_state.temp_target["task"].strip()
        )
        
        if st.button(
            "➕ Add Target", 
            key="add_target_btn", 
            type="primary",
            disabled=not can_add_target
        ):
            # Check for duplicates
            existing_tasks = [
                (t["category"], t["task"]) 
                for t in st.session_state.period_plan["targets"]
            ]
            
            new_combo = (
                st.session_state.temp_target["category"], 
                st.session_state.temp_target["task"].strip()
            )
            
            if new_combo not in existing_tasks:
                st.session_state.period_plan["targets"].append({
                    "category": st.session_state.temp_target["category"],
                    "task": st.session_state.temp_target["task"].strip(),
                    "daily_sessions": st.session_state.temp_target["daily_sessions"]
                })
                
                # Reset temp target
                st.session_state.temp_target = {
                    "category": "",
                    "task": "",
                    "daily_sessions": 1
                }
                
                st.success("✅ Target added!")
                st.rerun()
            else:
                st.error("⚠️ This category-task combination already exists")
        
        if not can_add_target:
            if not st.session_state.temp_target["category"]:
                st.caption("⚠️ Select category")
            elif not st.session_state.temp_target["task"].strip():
                st.caption("⚠️ Enter task")
    
    st.divider()
    
    # === PLAN SUMMARY ===
    if st.session_state.period_plan["targets"] and st.session_state.period_plan["name"].strip():
        st.markdown("#### 📊 Plan Summary")
        
        targets = st.session_state.period_plan["targets"]
        duration = st.session_state.period_plan["duration_days"]
        
        total_daily_sessions = sum(t["daily_sessions"] for t in targets)
        total_period_sessions = total_daily_sessions * duration
        total_hours = (total_period_sessions * 25) / 60
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📅 Daily Sessions", total_daily_sessions)
        
        with col2:
            st.metric("🎯 Total Sessions", total_period_sessions)
        
        with col3:
            st.metric("⏰ Total Hours", f"{total_hours:.1f}h")
        
        # Detailed breakdown
        st.markdown("**Target Breakdown:**")
        
        for target in targets:
            daily_minutes = target["daily_sessions"] * 25
            period_total = target["daily_sessions"] * duration
            
            st.markdown(f"""
            <div style="background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px; margin: 6px 0;">
                <div style="font-weight: bold; color: #1f2937; margin-bottom: 4px;">
                    🎯 {target['task']} <span style="color: #6b7280;">({target['category']})</span>
                </div>
                <div style="font-size: 0.9em; color: #6b7280;">
                    📅 {target['daily_sessions']} sessions/day ({daily_minutes}min/day) → 
                    🎯 {period_total} total sessions ({period_total * 25}min total)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # === CREATE PLAN BUTTON ===
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🚀 Create Plan", key="create_plan_btn", type="primary", use_container_width=True):
                # Validate plan
                if not st.session_state.period_plan["name"].strip():
                    st.error("⚠️ Please enter a plan name")
                elif not st.session_state.period_plan["targets"]:
                    st.error("⚠️ Please add at least one target")
                else:
                    # Save plan
                    save_period_target(
                        st.session_state.period_plan["name"].strip(),
                        st.session_state.period_plan["start_date"].isoformat(),
                        end_date.isoformat(),
                        st.session_state.period_plan["targets"],
                        st.session_state.user
                    )
                    
                    # Reset plan state
                    st.session_state.period_plan = {
                        "name": "",
                        "start_date": datetime.now(IST).date(),
                        "duration_days": 7,
                        "targets": [],
                        "editing_target": None
                    }
                    
                    st.session_state.temp_target = {
                        "category": "",
                        "task": "",
                        "daily_sessions": 1
                    }
                    
                    st.success("✅ Plan created successfully!")
                    st.rerun()
    
    else:
        # Call to action
        missing_items = []
        if not st.session_state.period_plan["name"].strip():
            missing_items.append("plan name")
        if not st.session_state.period_plan["targets"]:
            missing_items.append("targets")
        
        if missing_items:
            st.info(f"📝 Please add: {', '.join(missing_items)} to create your plan")
        
        # Show example or help
        with st.expander("💡 How to create a plan"):
            st.markdown("""
            **Steps to create your period target plan:**
            
            1. **📋 Plan Information**: Enter a descriptive name and select duration
            2. **🎯 Add Targets**: For each area you want to focus on:
               - Select or create a category (e.g., "Learning", "Projects")  
               - Enter specific task (e.g., "UGC NET Paper 1", "SQL Practice")
               - Set daily sessions target (how many 25min sessions per day)
            3. **📊 Review Summary**: Check your plan before creating
            4. **🚀 Create Plan**: Your plan will become active immediately
            
            **Example Plan: "UGC NET Preparation Week"**
            - 🎯 UGC NET Paper 1 (Learning): 1 session/day  
            - 🎯 UGC NET Paper 2 (Learning): 1 session/day
            - 🎯 SQL Projects (Development): 1 session/day
            - 🎯 Practice Tests (Practice): 1 session/day
            
            **Total**: 4 sessions/day (100 min/day) = 700 minutes/week
            """)
        
        # Quick start templates
        with st.expander("🚀 Quick Start Templates"):
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📚 Study Week Template", key="study_template"):
                    st.session_state.period_plan["name"] = "Study Week"
                    st.session_state.period_plan["duration_days"] = 7
                    st.session_state.period_plan["targets"] = [
                        {"category": "Learning", "task": "Main Subject", "daily_sessions": 2},
                        {"category": "Learning", "task": "Practice Problems", "daily_sessions": 1},
                        {"category": "Research", "task": "Additional Reading", "daily_sessions": 1}
                    ]
                    st.rerun()
            
            with col2:
                if st.button("💼 Project Sprint Template", key="project_template"):
                    st.session_state.period_plan["name"] = "Project Sprint"
                    st.session_state.period_plan["duration_days"] = 21
                    st.session_state.period_plan["targets"] = [
                        {"category": "Development", "task": "Core Features", "daily_sessions": 2},
                        {"category": "Development", "task": "Testing", "daily_sessions": 1},
                        {"category": "Planning", "task": "Documentation", "daily_sessions": 1}
                    ]
                    st.rerun()

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
    elif st.session_state.page == "📅 Period Targets":
        render_period_targets_page()
    elif st.session_state.page == "📝 Notes Saver":
        render_notes_saver_page()
    elif st.session_state.page == "📊 Analytics":
        render_analytics_page()
    elif st.session_state.page == "🗂️ Notes Viewer":
        render_notes_viewer_page()

if __name__ == "__main__":
    main()
