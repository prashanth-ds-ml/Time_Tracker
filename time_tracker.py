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
col_cat1, col_cat2 = st.columns([3, 1])

with col_cat1:
    cat_options = st.session_state.custom_categories + ["➕ Add New Category"]
    category_selection = st.selectbox("Select Category", cat_options)

with col_cat2:
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
    if st.button("🗑️ Manage"):
        with st.expander("Manage Categories", expanded=True):
            st.write("**Current Categories:**")
            for i, cat in enumerate(st.session_state.custom_categories):
                col_a, col_b = st.columns([4, 1])
                col_a.write(f"• {cat}")
                if col_b.button("🗑️", key=f"del_{i}"):
                    st.session_state.custom_categories.remove(cat)
                    st.rerun()

# Handle category selection/addition
if category_selection == "➕ Add New Category":
    new_cat = st.text_input("Enter New Category", placeholder="e.g., Reading, Exercise, Project X")
    if st.button("✅ Add Category") and new_cat:
        if new_cat not in st.session_state.custom_categories:
            st.session_state.custom_categories.append(new_cat)
            st.session_state.category = new_cat
            st.success(f"Added category: {new_cat}")
            st.rerun()
        else:
            st.warning("Category already exists!")
    st.session_state.category = new_cat if new_cat else ""
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
    df_work = df[df["pomodoro_type"] == "Work"]
    df_today = df[df["date"].dt.date == today]
    work_today = df_today[df_today["pomodoro_type"] == "Work"]
    break_today = df_today[df_today["pomodoro_type"] == "Break"]

    # === DAILY GOALS & PROGRESS ===
    if "daily_goal" not in st.session_state:
        st.session_state.daily_goal = 2  # Default 2 pomodoros (50 minutes)
    
    # Quick metrics row
    col1, col2, col3, col4 = st.columns(4)
    today_progress = len(work_today)
    today_minutes = work_today['duration'].sum()
    progress_pct = min(100, (today_progress / st.session_state.daily_goal) * 100)
    
    col1.metric("💼 Today's Work", f"{today_minutes} min")
    col2.metric("🎯 Progress", f"{today_progress}/{st.session_state.daily_goal}")
    col3.metric("☕ Breaks", f"{break_today['duration'].sum()} min")
    col4.metric("📈 Goal %", f"{progress_pct:.0f}%")

    # Progress bar with goal setting
    goal_col1, goal_col2 = st.columns([1, 3])
    with goal_col1:
        new_goal = st.number_input("Daily Goal", min_value=1, max_value=20, value=st.session_state.daily_goal)
        if new_goal != st.session_state.daily_goal:
            st.session_state.daily_goal = new_goal
    
    with goal_col2:
        st.progress(progress_pct / 100)
        if today_progress >= st.session_state.daily_goal:
            st.success("🎉 Daily goal achieved!")
        elif today_progress >= st.session_state.daily_goal * 0.75:
            st.info("💪 Almost there!")

    st.markdown("---")
    
    # === CORE VISUALIZATIONS ===
    
    # 1. DAILY OVERVIEW CHART
    st.subheader("📊 Daily Work Overview (Last 30 Days)")
    
    # Prepare daily data
    daily_work_counts = df_work.groupby(df_work["date"].dt.date).size()
    daily_minutes = df_work.groupby(df_work["date"].dt.date)["duration"].sum()
    
    # Last 30 days data
    daily_overview = []
    for i in range(30):
        check_date = today - timedelta(days=29-i)
        daily_count = daily_work_counts.get(check_date, 0)
        daily_min = daily_minutes.get(check_date, 0)
        daily_overview.append({
            'date': check_date,
            'date_str': check_date.strftime('%b %d'),
            'pomodoros': daily_count,
            'minutes': daily_min,
            'goal_met': daily_count >= st.session_state.daily_goal
        })
    
    overview_df = pd.DataFrame(daily_overview)
    
    # Create daily chart
    daily_fig = px.bar(
        overview_df,
        x='date_str',
        y='minutes',
        color='goal_met',
        title="Daily Work Minutes vs Goal",
        labels={'minutes': 'Minutes Worked', 'date_str': 'Date'},
        color_discrete_map={True: '#00FF00', False: '#FFA500'}
    )
    daily_fig.add_hline(y=st.session_state.daily_goal * 25, line_dash="dash", 
                       line_color="red", annotation_text=f"Goal ({st.session_state.daily_goal * 25} min)")
    st.plotly_chart(daily_fig, use_container_width=True)

    # 2. STREAK CALENDAR
    st.subheader("🔥 Consistency Calendar & Streaks")
    
    # Calculate streaks
    current_streak = 0
    max_streak = 0
    temp_streak = 0
    
    for i in range(365):
        check_date = today - timedelta(days=i)
        daily_count = daily_work_counts.get(check_date, 0)
        
        if daily_count >= st.session_state.daily_goal:
            temp_streak += 1
            max_streak = max(max_streak, temp_streak)
            if i == 0:
                current_streak = temp_streak
        else:
            if i == 0:
                current_streak = 0
            temp_streak = 0
    
    # Streak metrics
    streak_col1, streak_col2, streak_col3 = st.columns(3)
    streak_col1.metric("🔥 Current Streak", f"{current_streak} days")
    streak_col2.metric("🏆 Best Streak", f"{max_streak} days")
    if current_streak == 0:
        streak_col3.metric("💡 Next Goal", "Start today!")
    elif current_streak < 7:
        streak_col3.metric("💪 To 1 Week", f"{7-current_streak} days")
    else:
        streak_col3.metric("🌟 Status", "On fire!")
    
    # Calendar heatmap (last 6 weeks)
    calendar_data = []
    for i in range(42):
        check_date = today - timedelta(days=41-i)
        daily_count = daily_work_counts.get(check_date, 0)
        daily_min = daily_minutes.get(check_date, 0)
        
        if daily_min >= st.session_state.daily_goal * 25:
            intensity = 3
        elif daily_min >= 25:
            intensity = 2
        elif daily_min > 0:
            intensity = 1
        else:
            intensity = 0
        
        calendar_data.append({
            'date': check_date,
            'weekday': check_date.strftime('%a'),
            'week': i // 7,
            'pomodoros': daily_count,
            'minutes': daily_min,
            'intensity': intensity,
            'is_today': check_date == today
        })
    
    calendar_df = pd.DataFrame(calendar_data)
    
    # Create calendar heatmap
    calendar_fig = px.scatter(
        calendar_df,
        x='weekday',
        y='week',
        size='minutes',
        color='intensity',
        hover_data=['date', 'pomodoros', 'minutes'],
        title="6-Week Consistency Calendar",
        color_continuous_scale=[[0, '#LIGHTGRAY'], [0.33, '#FFA500'], [0.66, '#FFFF00'], [1, '#00FF00']],
        size_max=25
    )
    
    # Highlight today
    today_data = calendar_df[calendar_df['is_today']]
    if not today_data.empty:
        calendar_fig.add_scatter(
            x=today_data['weekday'],
            y=today_data['week'],
            mode='markers',
            marker=dict(size=30, color='red', symbol='circle-open', line=dict(width=3)),
            name='Today',
            showlegend=False
        )
    
    calendar_fig.update_layout(height=300, showlegend=False)
    st.plotly_chart(calendar_fig, use_container_width=True)
    
    # 3. TIME DISTRIBUTION
    st.subheader("🥧 Time Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category distribution
        category_time = df_work.groupby('category')['duration'].sum().reset_index()
        if not category_time.empty:
            cat_fig = px.pie(category_time, values='duration', names='category', title="Time by Category")
            st.plotly_chart(cat_fig, use_container_width=True)
    
    with col2:
        # Weekly pattern
        df_work_copy = df_work.copy()
        df_work_copy['weekday'] = df_work_copy['date'].dt.day_name()
        weekly_data = df_work_copy.groupby('weekday')['duration'].sum().reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        ).fillna(0).reset_index()
        
        week_fig = px.bar(weekly_data, x='weekday', y='duration', title="Weekly Pattern")
        week_fig.add_hline(y=st.session_state.daily_goal * 25, line_dash="dash", 
                          line_color="red", annotation_text="Goal")
        st.plotly_chart(week_fig, use_container_width=True)
    
    # 4. DETAILED BREAKDOWN TABLE
    st.subheader("📋 Category & Task Breakdown")
    
    # Show category and task breakdown
    breakdown = df_work.groupby(['category', 'task']).agg({
        'duration': ['sum', 'count'],
        'date': 'nunique'
    }).round(0)
    breakdown.columns = ['Total Minutes', 'Sessions', 'Days']
    breakdown = breakdown.sort_values('Total Minutes', ascending=False)
    breakdown['Hours'] = (breakdown['Total Minutes'] / 60).round(1)
    
    # Reorder columns for better display
    breakdown = breakdown[['Hours', 'Total Minutes', 'Sessions', 'Days']]
    st.dataframe(breakdown, use_container_width=True)
    
    # 5. KEY INSIGHTS
    st.subheader("💡 Key Insights")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        # Consistency score
        if len(daily_work_counts) >= 7:
            last_7_days = [daily_work_counts.get(today - timedelta(days=i), 0) for i in range(7)]
            consistency = len([d for d in last_7_days if d > 0]) / 7 * 100
            st.metric("🎯 Consistency", f"{consistency:.0f}%")
            if consistency >= 80:
                st.success("Excellent!")
            elif consistency >= 60:
                st.info("Good")
            else:
                st.warning("Need improvement")
    
    with insight_col2:
        # Best day
        if not df_work.empty:
            best_weekday = df_work.groupby(df_work['date'].dt.day_name())['duration'].sum().idxmax()
            st.metric("💪 Best Day", best_weekday)
            
            # Peak hour
            df_work_copy = df_work.copy()
            df_work_copy['hour'] = pd.to_datetime(df_work_copy['time'], format='%I:%M %p', errors='coerce').dt.hour
            if not df_work_copy['hour'].isna().all():
                peak_hour = df_work_copy.groupby('hour')['duration'].sum().idxmax()
                st.info(f"Peak: {peak_hour:02d}:00")
    
    with insight_col3:
        # Total stats
        total_minutes = df_work['duration'].sum()
        total_hours = total_minutes // 60
        st.metric("⏱️ Total Time", f"{total_hours}h {total_minutes%60}m")
        
        if len(daily_work_counts) >= 30:
            avg_daily = daily_work_counts.tail(30).mean()
            st.info(f"Avg: {avg_daily:.1f}/day")
    
    # Motivational messages
    st.markdown("---")
    if len(work_today) == 0:
        st.warning("🚀 **Start Today:** Just one Pomodoro to begin!")
    elif current_streak >= 7:
        st.success(f"🔥 **{current_streak}-day streak!** You're building an amazing habit!")
    elif current_streak >= 3:
        st.info("💪 **Building momentum!** Keep the streak alive!")
    
    # Goal achievement insights
    if len(daily_work_counts) >= 7:
        recent_avg = daily_work_counts.tail(7).mean()
        if recent_avg >= st.session_state.daily_goal:
            st.success("📈 **Crushing your goals!** Consider raising the bar.")
        elif recent_avg >= st.session_state.daily_goal * 0.75:
            st.info("🎯 **Close to your goal!** Small push needed.")
        else:
            gap = st.session_state.daily_goal - recent_avg
            st.info(f"💭 **{gap:.1f} more Pomodoros** needed to hit your daily target consistently.")

else:
    st.info("📝 No work sessions recorded yet. Start your first Pomodoro!")
    st.markdown("### 🚀 Getting Started Tips:")
    st.markdown("- Set a category and task above")
    st.markdown("- Click 'Start Work Pomodoro' to begin")
    st.markdown("- Analytics will appear after your first session")

