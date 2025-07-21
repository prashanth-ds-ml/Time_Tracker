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

    # === DAILY GOALS & STREAKS ===
    st.markdown("---")
    st.header("🎯 Daily Goals & Progress")
    
    # Goal setting
    if "daily_goal" not in st.session_state:
        st.session_state.daily_goal = 2  # Default 2 pomodoros (50 minutes)
    
    goal_col1, goal_col2, goal_col3 = st.columns(3)
    with goal_col1:
        new_goal = st.number_input("Daily Pomodoro Goal", min_value=1, max_value=20, value=st.session_state.daily_goal)
        if new_goal != st.session_state.daily_goal:
            st.session_state.daily_goal = new_goal
    
    with goal_col2:
        today_progress = len(work_today)
        today_minutes = work_today['duration'].sum()
        progress_pct = min(100, (today_progress / st.session_state.daily_goal) * 100)
        st.metric("Today's Progress", f"{today_progress}/{st.session_state.daily_goal}", 
                 f"{progress_pct:.0f}% complete")
    
    with goal_col3:
        st.metric("Minutes Today", f"{today_minutes} min", 
                 f"Target: {st.session_state.daily_goal * 25} min")
    
    # Progress bar
    st.progress(progress_pct / 100)
    if today_progress >= st.session_state.daily_goal:
        st.success("🎉 Daily goal achieved! You're crushing it!")
    elif today_progress >= st.session_state.daily_goal * 0.75:
        st.info("💪 Almost there! Keep pushing!")
    elif today_minutes >= 25:
        st.info("⚡ Good start! Keep the momentum going!")
    
    st.markdown("---")
    st.header("🔥 Streak Analytics")
    
    # Fixed streak calculation
    daily_work_counts = df_work.groupby(df_work["date"].dt.date).size()
    
    # Calculate streaks properly
    current_streak = 0
    max_streak = 0
    temp_streak = 0
    
    # Check last 365 days for comprehensive streak tracking
    for i in range(365):
        check_date = today - timedelta(days=i)
        daily_count = daily_work_counts.get(check_date, 0)
        
        if daily_count >= st.session_state.daily_goal:
            temp_streak += 1
            max_streak = max(max_streak, temp_streak)
            if i == 0:  # Today
                current_streak = temp_streak
        else:
            if i == 0:
                current_streak = 0
            temp_streak = 0
    
    streak_col1, streak_col2, streak_col3 = st.columns(3)
    streak_col1.metric("🔥 Current Streak", f"{current_streak} day(s)")
    streak_col2.metric("🏆 Best Streak", f"{max_streak} day(s)")
    
    # Streak motivation
    if current_streak == 0:
        streak_col3.metric("💡 Motivation", "Start today!")
    elif current_streak < 7:
        streak_col3.metric("💪 Keep Going", f"{7-current_streak} to week!")
    else:
        streak_col3.metric("🌟 Amazing!", "On fire!")
    
    # === ENHANCED VISUALIZATIONS ===
    st.markdown("---")
    st.header("📈 Time Analysis & Insights")
    
    # 1. Weekly heatmap
    st.subheader("📅 Weekly Activity Heatmap")
    df_work_copy = df_work.copy()
    df_work_copy['weekday'] = df_work_copy['date'].dt.day_name()
    df_work_copy['week'] = df_work_copy['date'].dt.isocalendar().week
    
    weekly_data = df_work_copy.groupby(['week', 'weekday'])['duration'].sum().reset_index()
    
    if not weekly_data.empty:
        heatmap_fig = px.density_heatmap(
            weekly_data, 
            x='weekday', 
            y='week',
            z='duration',
            title="Weekly Activity Pattern (Minutes per Day)",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # 2. Calendar view for streak visualization
    st.subheader("📅 Consistency Calendar (Last 60 Days)")
    
    # Create calendar data
    calendar_data = []
    for i in range(60):
        check_date = today - timedelta(days=59-i)
        daily_count = daily_work_counts.get(check_date, 0)
        daily_minutes = df_work[df_work['date'].dt.date == check_date]['duration'].sum()
        
        # Define consistency levels
        if daily_minutes >= 50:  # 2 pomodoros = 50 minutes
            status = "🔥 Perfect"
            color = "green"
        elif daily_minutes >= 25:  # 1 pomodoro = 25 minutes
            status = "💪 Good"
            color = "orange"
        elif daily_minutes > 0:
            status = "⚡ Started"
            color = "yellow"
        else:
            status = "❌ Miss"
            color = "lightgray"
        
        calendar_data.append({
            'date': check_date,
            'minutes': daily_minutes,
            'pomodoros': daily_count,
            'status': status,
            'color': color,
            'week': check_date.isocalendar()[1],
            'weekday': check_date.strftime('%a'),
            'day': check_date.day
        })
    
    calendar_df = pd.DataFrame(calendar_data)
    
    # Create calendar visualization
    if not calendar_df.empty:
        calendar_fig = px.scatter(
            calendar_df,
            x='weekday',
            y='week',
            size='minutes',
            color='minutes',
            hover_data=['date', 'pomodoros', 'status'],
            title="Consistency Calendar (Bubble size = minutes worked)",
            color_continuous_scale="RdYlGn",
            size_max=20
        )
        calendar_fig.update_layout(height=400)
        st.plotly_chart(calendar_fig, use_container_width=True)
        
        # Consistency legend
        st.markdown("""
        **Legend:** 🔥 Perfect (50+ min) | 💪 Good (25-49 min) | ⚡ Started (<25 min) | ❌ Miss (0 min)
        """)
    
    # 3. Enhanced time distribution by category AND task
    st.subheader("🥧 Time Distribution by Category & Tasks")
    if not df_work.empty:
        # Category distribution
        col1, col2 = st.columns(2)
        
        with col1:
            category_time = df_work.groupby('category')['duration'].sum().reset_index()
            cat_pie_fig = px.pie(
                category_time, 
                values='duration', 
                names='category',
                title="Time by Category"
            )
            st.plotly_chart(cat_pie_fig, use_container_width=True)
        
        with col2:
            # Combined category + task distribution
            df_work['category_task'] = df_work['category'] + ' → ' + df_work['task']
            task_time = df_work.groupby('category_task')['duration'].sum().sort_values(ascending=False).head(10)
            
            if not task_time.empty:
                task_pie_fig = px.pie(
                    values=task_time.values,
                    names=task_time.index,
                    title="Top 10 Category → Task Combinations"
                )
                st.plotly_chart(task_pie_fig, use_container_width=True)
        
        # Detailed breakdown table
        st.subheader("📊 Detailed Time Breakdown")
        detailed_breakdown = df_work.groupby(['category', 'task']).agg({
            'duration': ['sum', 'count'],
            'date': 'nunique'
        }).round(2)
        
        detailed_breakdown.columns = ['Total Minutes', 'Sessions', 'Days Worked']
        detailed_breakdown = detailed_breakdown.sort_values('Total Minutes', ascending=False)
        detailed_breakdown['Hours'] = (detailed_breakdown['Total Minutes'] / 60).round(1)
        
        st.dataframe(detailed_breakdown, use_container_width=True)
    
    # 3. Performance trends
    st.subheader("📊 Performance Trends (Last 30 Days)")
    cutoff_date = datetime.now(IST).date() - timedelta(days=30)
    last_30_days = df_work[df_work['date'].dt.date >= cutoff_date]
    
    if not last_30_days.empty:
        trend_data = last_30_days.groupby(last_30_days['date'].dt.date).agg({
            'duration': 'sum',
            'task': 'count'
        }).reset_index()
        trend_data.columns = ['date', 'total_minutes', 'pomodoro_count']
        trend_data['efficiency'] = trend_data['total_minutes'] / trend_data['pomodoro_count']
        
        # Dual axis chart
        trend_fig = px.line(
            trend_data, 
            x='date', 
            y='pomodoro_count',
            title="Daily Pomodoro Count vs Goal",
            labels={'pomodoro_count': 'Pomodoros Completed'}
        )
        trend_fig.add_hline(y=st.session_state.daily_goal, line_dash="dash", 
                           line_color="red", annotation_text="Daily Goal")
        st.plotly_chart(trend_fig, use_container_width=True)
    
    # === PSYCHOLOGY-DRIVEN INSIGHTS ===
    st.markdown("---")
    st.header("🧠 Psychology Insights & Motivation")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.subheader("📈 Consistency Score")
        if len(daily_work_counts) >= 7:
            last_7_days = [daily_work_counts.get(today - timedelta(days=i), 0) 
                          for i in range(7)]
            consistency = len([d for d in last_7_days if d > 0]) / 7 * 100
            st.metric("7-Day Consistency", f"{consistency:.0f}%")
            
            if consistency >= 80:
                st.success("🌟 Excellent consistency! You're building a strong habit.")
            elif consistency >= 60:
                st.info("👍 Good consistency! Try to fill in the gaps.")
            else:
                st.warning("⚠️ Focus on consistency. Small daily wins compound!")
        else:
            st.info("Complete 7 days to see your consistency score.")
    
    with insight_col2:
        st.subheader("⚡ Peak Performance")
        if not df_work.empty:
            df_work_copy = df_work.copy()
            df_work_copy['hour'] = pd.to_datetime(df_work_copy['time'], format='%I:%M %p').dt.hour
            peak_hour = df_work_copy.groupby('hour')['duration'].sum().idxmax()
            peak_time = f"{peak_hour:02d}:00"
            st.metric("Most Productive Hour", peak_time)
            st.info(f"💡 Schedule important tasks around {peak_time} for maximum efficiency!")
    
    # Motivational insights
    st.subheader("💡 Personalized Insights")
    
    if len(work_today) == 0:
        st.warning("🚀 **Getting Started Tip:** Just complete 1 Pomodoro today. The hardest part is starting!")
    elif current_streak >= 7:
        st.success(f"🔥 **Streak Master:** {current_streak} days strong! You're in the top 10% of consistent users.")
    elif current_streak >= 3:
        st.info("💪 **Building Momentum:** Great streak! Research shows it takes 21 days to form a habit.")
    
    # Goal achievement prediction
    if len(daily_work_counts) >= 7:
        avg_daily = daily_work_counts.tail(7).mean()
        if avg_daily >= st.session_state.daily_goal:
            st.success("📈 **Goal Achiever:** You're consistently meeting your goals! Consider raising the bar.")
        else:
            gap = st.session_state.daily_goal - avg_daily
            st.info(f"🎯 **Goal Gap:** You're {gap:.1f} Pomodoros away from your daily goal on average. Small adjustments lead to big wins!")
    
    # Weekly reflection
    if datetime.now(IST).weekday() == 6:  # Sunday
        week_total = daily_work_counts.get(today, 0)
        for i in range(1, 7):
            week_total += daily_work_counts.get(today - timedelta(days=i), 0)
        
        st.subheader("📝 Weekly Reflection")
        st.info(f"This week you completed {week_total} Pomodoros. What went well? What can you improve next week?")
else:
    st.info("No log records found in MongoDB.")

