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

    # === PSYCHOLOGY-DRIVEN PROGRESSIVE GOALS ===
    
    # Calculate user's journey days (total days with any activity)
    active_days = len(df_work.groupby(df_work["date"].dt.date).size())
    
    # Progressive goal system based on habit formation science
    def get_adaptive_goal(days_active):
        if days_active <= 5:
            return 1, "🌱 Building Habit", "Start small, be consistent"
        elif days_active <= 12:
            return 2, "🔥 Establishing Routine", "Building momentum"
        elif days_active <= 19:
            return 3, "💪 Strengthening Discipline", "Pushing boundaries"
        else:
            return 4, "🚀 Peak Performance", "Maintaining excellence"
    
    adaptive_goal, phase_name, phase_desc = get_adaptive_goal(active_days)
    min_streak_goal = 2 if active_days > 12 else adaptive_goal  # Minimum for streak after day 12
    
    # === TOP SECTION: DAILY GOALS & PROGRESS ===
    st.markdown("""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 15px; margin-bottom: 20px;'>
        <h2 style='color: white; text-align: center; margin: 0;'>🎯 Today's Mission</h2>
    </div>
    """, unsafe_allow_html=True)
    
    today_progress = len(work_today)
    today_minutes = work_today['duration'].sum()
    progress_pct = min(100, (today_progress / adaptive_goal) * 100)
    
    # Progress metrics in a clean layout
    progress_col1, progress_col2, progress_col3 = st.columns([2, 1, 1])
    
    with progress_col1:
        st.markdown(f"""
        <div style='background: white; padding: 15px; border-radius: 10px; text-align: center;'>
            <h3 style='margin: 0; color: #333;'>{phase_name}</h3>
            <p style='margin: 5px 0; color: #666;'>{phase_desc}</p>
            <div style='background: #f0f0f0; border-radius: 20px; height: 30px; margin: 10px 0;'>
                <div style='background: linear-gradient(90deg, #4CAF50, #45a049); 
                           width: {progress_pct}%; height: 100%; border-radius: 20px; 
                           display: flex; align-items: center; justify-content: center;'>
                    <span style='color: white; font-weight: bold;'>{today_progress}/{adaptive_goal}</span>
                </div>
            </div>
            <p style='margin: 0; font-size: 14px; color: #888;'>Day {active_days} of your journey</p>
        </div>
        """, unsafe_allow_html=True)
    
    with progress_col2:
        st.metric("⏱️ Minutes", f"{today_minutes}")
        st.metric("☕ Breaks", f"{break_today['duration'].sum()}")
    
    with progress_col3:
        st.metric("📅 Active Days", f"{active_days}")
        if today_progress >= adaptive_goal:
            st.success("✨ Goal Achieved!")
        elif today_progress >= adaptive_goal * 0.5:
            st.info(f"🎯 {adaptive_goal - today_progress} to go!")
        else:
            st.warning("🚀 Let's start!")

    st.markdown("---")
    
    # === CORE VISUALIZATIONS (4 KEY CHARTS) ===
    
    # 1. DAILY PERFORMANCE CHART
    st.subheader("📈 Daily Performance Trends")
    
    # Prepare daily data with adaptive goals
    daily_work_counts = df_work.groupby(df_work["date"].dt.date).size()
    daily_minutes = df_work.groupby(df_work["date"].dt.date)["duration"].sum()
    
    # Last 30 days with adaptive goal tracking
    daily_overview = []
    for i in range(30):
        check_date = today - timedelta(days=29-i)
        day_num = active_days - (29-i) if active_days > (29-i) else 1
        day_goal, _, _ = get_adaptive_goal(day_num)
        
        daily_count = daily_work_counts.get(check_date, 0)
        daily_min = daily_minutes.get(check_date, 0)
        
        daily_overview.append({
            'date': check_date,
            'date_str': check_date.strftime('%m/%d'),
            'pomodoros': daily_count,
            'minutes': daily_min,
            'goal': day_goal * 25,
            'achieved': daily_count >= day_goal,
            'week': check_date.strftime('%U')
        })
    
    overview_df = pd.DataFrame(daily_overview)
    
    # Clean daily performance chart
    daily_fig = px.bar(
        overview_df,
        x='date_str',
        y='minutes',
        color='achieved',
        title="📊 Daily Minutes vs Progressive Goals",
        labels={'minutes': 'Minutes Worked', 'date_str': 'Date'},
        color_discrete_map={True: '#4CAF50', False: '#FF9800'},
        height=400
    )
    
    # Add goal line
    daily_fig.add_scatter(
        x=overview_df['date_str'],
        y=overview_df['goal'],
        mode='lines',
        name='Adaptive Goal',
        line=dict(color='red', dash='dash', width=2)
    )
    
    daily_fig.update_layout(
        showlegend=True,
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(daily_fig, use_container_width=True)

    # 2. CATEGORY & TASK HEATMAP
    st.subheader("🎯 Category & Task Focus Heatmap")
    
    if len(df_work) > 0:
        # Create category-task matrix
        task_category_matrix = df_work.groupby(['category', 'task'])['duration'].sum().reset_index()
        task_category_pivot = task_category_matrix.pivot(index='category', columns='task', values='duration').fillna(0)
        
        if not task_category_pivot.empty:
            heatmap_fig = px.imshow(
                task_category_pivot.values,
                labels=dict(x="Tasks", y="Categories", color="Minutes"),
                x=task_category_pivot.columns,
                y=task_category_pivot.index,
                color_continuous_scale="Blues",
                title="🎯 Time Investment Heatmap",
                height=400
            )
            heatmap_fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # 3. STREAK TRACKING
    st.subheader("🔥 Consistency Streak")
    
    # Calculate streaks with adaptive goals
    current_streak = 0
    max_streak = 0
    temp_streak = 0
    
    for i in range(365):
        check_date = today - timedelta(days=i)
        daily_count = daily_work_counts.get(check_date, 0)
        day_goal = min_streak_goal  # Use minimum streak requirement
        
        if daily_count >= day_goal:
            temp_streak += 1
            max_streak = max(max_streak, temp_streak)
            if i == 0:
                current_streak = temp_streak
        else:
            if i == 0:
                current_streak = 0
            temp_streak = 0
    
    # Streak metrics with psychology insights
    streak_col1, streak_col2, streak_col3 = st.columns(3)
    
    with streak_col1:
        st.metric("🔥 Current Streak", f"{current_streak} days")
        if current_streak >= 21:
            st.success("🎉 Habit Formed!")
        elif current_streak >= 7:
            st.info("💪 Building Strong!")
        elif current_streak >= 3:
            st.warning("🌱 Getting There!")
    
    with streak_col2:
        st.metric("🏆 Best Streak", f"{max_streak} days")
        if max_streak >= 30:
            st.success("🏆 Champion!")
        elif max_streak >= 14:
            st.info("⭐ Strong!")
    
    with streak_col3:
        next_milestone = 7 if current_streak < 7 else 21 if current_streak < 21 else 30
        days_to_go = next_milestone - current_streak if current_streak < next_milestone else 0
        if days_to_go > 0:
            st.metric("🎯 Next Milestone", f"{days_to_go} days to {next_milestone}")
        else:
            st.metric("🌟 Status", "Peak Performer!")
    
    # 4. WEEKLY PERFORMANCE PATTERN
    st.subheader("📅 Weekly Performance Pattern")
    
    df_work_copy = df_work.copy()
    df_work_copy['weekday'] = df_work_copy['date'].dt.day_name()
    weekly_data = df_work_copy.groupby('weekday')['duration'].sum().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    ).fillna(0).reset_index()
    
    week_fig = px.bar(
        weekly_data, 
        x='weekday', 
        y='duration', 
        title="📊 Weekly Focus Pattern",
        color='duration',
        color_continuous_scale='Viridis',
        height=400
    )
    week_fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    st.plotly_chart(week_fig, use_container_width=True)
    
    st.markdown("---")
    
    # === PSYCHOLOGY-DRIVEN INSIGHTS ===
    st.subheader("🧠 Psychology Insights & Habit Formation")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        # Habit formation stage
        if active_days <= 5:
            st.info("🌱 **Formation Stage**\nSmall steps build big habits")
        elif active_days <= 21:
            progress = (active_days / 21) * 100
            st.warning(f"🔥 **Building Stage**\n{progress:.0f}% to automatic habit")
        else:
            st.success("🎯 **Mastery Stage**\nHabit is becoming automatic!")
        
        # Consistency score
        if len(daily_work_counts) >= 7:
            last_7_days = [daily_work_counts.get(today - timedelta(days=i), 0) >= min_streak_goal for i in range(7)]
            consistency = sum(last_7_days) / 7 * 100
            st.metric("📈 Weekly Consistency", f"{consistency:.0f}%")
    
    with insight_col2:
        # Best performance day
        if not df_work.empty:
            best_weekday = df_work.groupby(df_work['date'].dt.day_name())['duration'].sum().idxmax()
            st.metric("💪 Peak Day", best_weekday)
            
            # Total achievement
            total_minutes = df_work['duration'].sum()
            total_hours = total_minutes // 60
            st.metric("⏱️ Total Focus", f"{total_hours}h {total_minutes%60}m")
    
    with insight_col3:
        # Achievement level
        if current_streak >= 30:
            st.success("🏆 **Master Level**\nYou've formed a strong habit!")
        elif current_streak >= 21:
            st.info("🌟 **Expert Level**\nHabit is nearly automatic!")
        elif current_streak >= 7:
            st.warning("📈 **Intermediate Level**\nBuilding strong momentum!")
        elif current_streak >= 3:
            st.info("🌱 **Beginner Level**\nGreat start, keep going!")
        else:
            st.warning("🚀 **Ready to Start**\nBegin your journey today!")
    
    # === MOTIVATIONAL INSIGHTS ===
    st.markdown("---")
    st.subheader("💡 Personalized Coaching")
    
    # Science-backed motivational messages
    if active_days == 1:
        st.info("🎉 **Day 1 Complete!** Research shows it takes 21 days to form a habit. You've started!")
    elif active_days == 7:
        st.success("🔥 **Week 1 Done!** Your brain is already creating new neural pathways. Keep building!")
    elif active_days == 21:
        st.success("🏆 **21 Days!** Congratulations! Your habit is becoming automatic. The hardest part is behind you!")
    elif active_days == 66:
        st.success("👑 **66 Days!** You've reached the average time for habit automation. You're a productivity master!")
    
    # Adaptive encouragement
    if today_progress == 0:
        if current_streak > 0:
            st.warning(f"🔥 **Don't break the {current_streak}-day streak!** Just {adaptive_goal} Pomodoro(s) to keep it alive.")
        else:
            st.info(f"🚀 **Start small:** Just {adaptive_goal} Pomodoro(s) today. Progress beats perfection!")
    elif today_progress >= adaptive_goal:
        if adaptive_goal < 4:
            st.success(f"🎯 **Goal achieved!** Ready for the next level? Your brain can handle {adaptive_goal + 1} Pomodoros.")
        else:
            st.success("🏆 **Peak performance achieved!** You're operating at your optimal capacity.")
    
    # Personalized recommendations
    if len(daily_work_counts) >= 14:
        avg_weekly = daily_work_counts.tail(14).mean()
        if avg_weekly < adaptive_goal * 0.7:
            st.info(f"💡 **Tip:** Try scheduling Pomodoros at consistent times. Routine strengthens habit formation.")
        elif current_streak < 7:
            st.info("💪 **Focus on consistency** over intensity. Small daily wins build lasting habits.")

else:
    st.info("📝 No work sessions recorded yet. Start your first Pomodoro!")
    st.markdown("### 🚀 Getting Started Tips:")
    st.markdown("- Set a category and task above")
    st.markdown("- Click 'Start Work Pomodoro' to begin")
    st.markdown("- Analytics will appear after your first session")

