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
    st.header("📈 Advanced Analytics & Insights")
    
    # 1. Productivity Score Dashboard
    st.subheader("🎯 Productivity Score & Performance Index")
    
    # Calculate productivity metrics
    recent_7_days = [daily_work_counts.get(today - timedelta(days=i), 0) for i in range(7)]
    recent_30_days = [daily_work_counts.get(today - timedelta(days=i), 0) for i in range(30)]
    
    consistency_7d = len([d for d in recent_7_days if d > 0]) / 7 * 100
    avg_daily_7d = sum(recent_7_days) / 7
    goal_achievement_7d = len([d for d in recent_7_days if d >= st.session_state.daily_goal]) / 7 * 100
    
    # Productivity score (weighted combination)
    productivity_score = (consistency_7d * 0.4 + goal_achievement_7d * 0.4 + min(avg_daily_7d/st.session_state.daily_goal * 100, 100) * 0.2)
    
    score_col1, score_col2, score_col3, score_col4 = st.columns(4)
    score_col1.metric("📊 Productivity Score", f"{productivity_score:.0f}/100", 
                     delta=f"{productivity_score-75:.0f}" if productivity_score > 75 else None)
    score_col2.metric("🎯 Goal Achievement", f"{goal_achievement_7d:.0f}%")
    score_col3.metric("⚡ Daily Average", f"{avg_daily_7d:.1f}")
    score_col4.metric("🔥 Consistency", f"{consistency_7d:.0f}%")
    
    # Performance gauge
    import plotly.graph_objects as go
    gauge_fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = productivity_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Productivity Index"},
        delta = {'reference': 75, 'increasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 90], 'color': "lightgreen"},
                {'range': [90, 100], 'color': "green"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    gauge_fig.update_layout(height=300)
    st.plotly_chart(gauge_fig, use_container_width=True)
    
    # 2. Time of Day Performance Analysis
    st.subheader("⏰ Peak Performance Times")
    if not df_work.empty:
        df_work_copy = df_work.copy()
        df_work_copy['hour'] = pd.to_datetime(df_work_copy['time'], format='%I:%M %p', errors='coerce').dt.hour
        df_work_copy = df_work_copy.dropna(subset=['hour'])
        
        if not df_work_copy.empty:
            hourly_performance = df_work_copy.groupby('hour').agg({
                'duration': ['sum', 'count', 'mean']
            }).round(2)
            hourly_performance.columns = ['Total Minutes', 'Sessions', 'Avg Duration']
            hourly_performance = hourly_performance.reset_index()
            
            # Create dual-axis chart
            perf_fig = px.bar(hourly_performance, x='hour', y='Total Minutes', 
                             title="Productivity by Hour of Day",
                             labels={'hour': 'Hour (24h format)', 'Total Minutes': 'Total Minutes Worked'})
            
            # Add sessions as line
            perf_fig.add_scatter(x=hourly_performance['hour'], y=hourly_performance['Sessions']*5, 
                               mode='lines+markers', name='Sessions (×5)', yaxis='y')
            
            st.plotly_chart(perf_fig, use_container_width=True)
            
            # Show insights
            best_hour = hourly_performance.loc[hourly_performance['Total Minutes'].idxmax(), 'hour']
            most_sessions_hour = hourly_performance.loc[hourly_performance['Sessions'].idxmax(), 'hour']
            
            insight_col1, insight_col2 = st.columns(2)
            insight_col1.info(f"🏆 **Peak Productivity:** {best_hour:02d}:00 - {best_hour+1:02d}:00")
            insight_col2.info(f"🔥 **Most Active:** {most_sessions_hour:02d}:00 - {most_sessions_hour+1:02d}:00")
    
    # 3. Weekly Pattern Analysis
    st.subheader("📅 Weekly Performance Pattern")
    df_work_copy = df_work.copy()
    df_work_copy['weekday'] = df_work_copy['date'].dt.day_name()
    df_work_copy['weekday_num'] = df_work_copy['date'].dt.dayofweek
    
    weekly_pattern = df_work_copy.groupby(['weekday', 'weekday_num']).agg({
        'duration': ['sum', 'count', 'mean']
    }).round(2)
    weekly_pattern.columns = ['Total Minutes', 'Sessions', 'Avg Session']
    weekly_pattern = weekly_pattern.reset_index().sort_values('weekday_num')
    
    # Create comprehensive weekly chart
    week_fig = px.bar(weekly_pattern, x='weekday', y='Total Minutes',
                     title="Weekly Performance Pattern",
                     color='Sessions', color_continuous_scale='Blues')
    
    # Add goal line
    weekly_goal = st.session_state.daily_goal * 25  # 25 min per pomodoro
    week_fig.add_hline(y=weekly_goal, line_dash="dash", line_color="red", 
                      annotation_text=f"Daily Goal ({weekly_goal} min)")
    
    st.plotly_chart(week_fig, use_container_width=True)
    
    # Weekly insights
    best_day = weekly_pattern.loc[weekly_pattern['Total Minutes'].idxmax(), 'weekday']
    worst_day = weekly_pattern.loc[weekly_pattern['Total Minutes'].idxmin(), 'weekday']
    
    week_col1, week_col2, week_col3 = st.columns(3)
    week_col1.success(f"🌟 **Strongest Day:** {best_day}")
    week_col2.warning(f"💪 **Growth Day:** {worst_day}")
    week_col3.info(f"📊 **Weekend vs Weekday:** {weekly_pattern.iloc[-2:]['Total Minutes'].mean():.0f} vs {weekly_pattern.iloc[:-2]['Total Minutes'].mean():.0f}")
    
    # 4. Focus Deep Dive Analysis
    st.subheader("🧠 Focus & Distraction Analysis")
    
    if not df_work.empty:
        # Session length analysis
        focus_col1, focus_col2 = st.columns(2)
        
        with focus_col1:
            # Analyze session clustering (how often sessions are back-to-back)
            df_work_sorted = df_work.sort_values(['date', 'time'])
            df_work_sorted['session_gap'] = df_work_sorted.groupby(df_work_sorted['date'].dt.date)['time'].shift(-1)
            
            # Focus session distribution
            session_dist = df_work.groupby(df_work['date'].dt.date).size()
            
            focus_fig = px.histogram(x=session_dist.values, nbins=10,
                                   title="Focus Sessions Distribution",
                                   labels={'x': 'Sessions per Day', 'y': 'Number of Days'})
            focus_fig.add_vline(x=st.session_state.daily_goal, line_dash="dash", 
                               line_color="red", annotation_text="Goal")
            st.plotly_chart(focus_fig, use_container_width=True)
        
        with focus_col2:
            # Category switching analysis (shows focus vs multitasking)
            daily_categories = df_work.groupby(df_work['date'].dt.date)['category'].nunique()
            
            switch_fig = px.histogram(x=daily_categories.values, nbins=8,
                                    title="Daily Category Switching",
                                    labels={'x': 'Categories per Day', 'y': 'Number of Days'})
            st.plotly_chart(switch_fig, use_container_width=True)
            
            avg_categories = daily_categories.mean()
            if avg_categories <= 2:
                st.success("🎯 **High Focus:** You typically stick to 1-2 categories daily!")
            elif avg_categories <= 3:
                st.info("⚖️ **Balanced:** Good balance between focus and variety.")
            else:
                st.warning("🔄 **High Switching:** Consider reducing daily category switching for better focus.")
    
    # 5. Progress Momentum Chart
    st.subheader("🚀 Progress Momentum & Trends")
    
    # Calculate 7-day rolling averages
    if len(daily_work_counts) >= 7:
        last_60_days_data = []
        for i in range(60):
            check_date = today - timedelta(days=59-i)
            daily_count = daily_work_counts.get(check_date, 0)
            last_60_days_data.append({
                'date': check_date,
                'sessions': daily_count,
                'minutes': daily_count * 25
            })
        
        momentum_df = pd.DataFrame(last_60_days_data)
        momentum_df['rolling_avg'] = momentum_df['sessions'].rolling(window=7, center=True).mean()
        momentum_df['trend'] = momentum_df['rolling_avg'].diff()
        
        # Create momentum chart
        momentum_fig = px.line(momentum_df, x='date', y='sessions', 
                              title="Daily Sessions with 7-Day Trend",
                              labels={'sessions': 'Daily Sessions', 'date': 'Date'})
        
        momentum_fig.add_scatter(x=momentum_df['date'], y=momentum_df['rolling_avg'], 
                               mode='lines', name='7-Day Average', line=dict(color='red', width=3))
        
        momentum_fig.add_hline(y=st.session_state.daily_goal, line_dash="dash", 
                              line_color="green", annotation_text="Goal")
        
        st.plotly_chart(momentum_fig, use_container_width=True)
        
        # Momentum insights
        recent_trend = momentum_df['trend'].tail(7).mean()
        if recent_trend > 0.1:
            st.success("📈 **Upward Momentum:** You're trending upward! Keep the energy!")
        elif recent_trend < -0.1:
            st.warning("📉 **Downward Trend:** Time to refocus and rebuild momentum.")
        else:
            st.info("➡️ **Steady State:** Consistent performance, consider pushing for growth.")
    
    # 6. Category Performance Matrix
    st.subheader("📊 Category Performance Matrix")
    if not df_work.empty and len(df_work['category'].unique()) > 1:
        category_analysis = df_work.groupby('category').agg({
            'duration': ['sum', 'count', 'mean'],
            'date': 'nunique'
        }).round(2)
        category_analysis.columns = ['Total Minutes', 'Sessions', 'Avg Duration', 'Days Active']
        category_analysis['Efficiency'] = category_analysis['Total Minutes'] / category_analysis['Sessions']
        category_analysis['Consistency'] = category_analysis['Sessions'] / category_analysis['Days Active']
        category_analysis = category_analysis.reset_index()
        
        # Create performance matrix scatter plot
        matrix_fig = px.scatter(category_analysis, x='Efficiency', y='Consistency',
                               size='Total Minutes', color='Sessions',
                               hover_data=['category'], title="Category Performance Matrix",
                               labels={'Efficiency': 'Minutes per Session', 
                                      'Consistency': 'Sessions per Active Day'})
        
        # Add quadrant lines
        avg_efficiency = category_analysis['Efficiency'].mean()
        avg_consistency = category_analysis['Consistency'].mean()
        
        matrix_fig.add_hline(y=avg_consistency, line_dash="dot", line_color="gray")
        matrix_fig.add_vline(x=avg_efficiency, line_dash="dot", line_color="gray")
        
        st.plotly_chart(matrix_fig, use_container_width=True)
        
        # Performance insights
        top_category = category_analysis.loc[category_analysis['Total Minutes'].idxmax(), 'category']
        most_consistent = category_analysis.loc[category_analysis['Consistency'].idxmax(), 'category']
        most_efficient = category_analysis.loc[category_analysis['Efficiency'].idxmax(), 'category']
        
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        perf_col1.metric("🏆 Most Time", top_category)
        perf_col2.metric("🎯 Most Consistent", most_consistent)
        perf_col3.metric("⚡ Most Efficient", most_efficient)
    
    # 7. Smart Calendar View with Streak Visualization
    st.subheader("📅 Smart Consistency Calendar")
    
    # Create enhanced calendar data
    calendar_data = []
    for i in range(42):  # 6 weeks view
        check_date = today - timedelta(days=41-i)
        daily_count = daily_work_counts.get(check_date, 0)
        daily_minutes = df_work[df_work['date'].dt.date == check_date]['duration'].sum()
        
        # Enhanced status levels
        if daily_minutes >= st.session_state.daily_goal * 25:
            status = "🔥 Goal Crushed"
            intensity = 4
            color = "#00FF00"
        elif daily_minutes >= 50:
            status = "💪 Strong Day"
            intensity = 3
            color = "#90EE90"
        elif daily_minutes >= 25:
            status = "⚡ Good Start"
            intensity = 2
            color = "#FFFF00"
        elif daily_minutes > 0:
            status = "👶 Baby Steps"
            intensity = 1
            color = "#FFA500"
        else:
            status = "❌ Rest Day"
            intensity = 0
            color = "#LIGHTGRAY"
        
        calendar_data.append({
            'date': check_date,
            'date_str': check_date.strftime('%b %d'),
            'minutes': daily_minutes,
            'pomodoros': daily_count,
            'status': status,
            'intensity': intensity,
            'color': color,
            'week': check_date.isocalendar()[1] - today.isocalendar()[1] + 6,
            'weekday': check_date.strftime('%a'),
            'day': check_date.day,
            'is_today': check_date == today,
            'is_weekend': check_date.weekday() >= 5
        })
    
    calendar_df = pd.DataFrame(calendar_data)
    
    # Create interactive calendar heatmap
    if not calendar_df.empty:
        # Custom color scale based on intensity
        calendar_fig = px.scatter(
            calendar_df,
            x='weekday',
            y='week',
            size='minutes',
            color='intensity',
            hover_data={
                'date_str': True,
                'pomodoros': True,
                'status': True,
                'minutes': True,
                'weekday': False,
                'week': False,
                'intensity': False
            },
            title="6-Week Consistency Heatmap (Hover for details)",
            color_continuous_scale=[[0, '#LIGHTGRAY'], [0.25, '#FFA500'], 
                                   [0.5, '#FFFF00'], [0.75, '#90EE90'], [1, '#00FF00']],
            size_max=30
        )
        
        # Highlight today
        today_data = calendar_df[calendar_df['is_today']]
        if not today_data.empty:
            calendar_fig.add_scatter(
                x=today_data['weekday'],
                y=today_data['week'],
                mode='markers',
                marker=dict(size=35, color='red', symbol='circle-open', line=dict(width=3)),
                name='Today',
                showlegend=False
            )
        
        calendar_fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(calendar_fig, use_container_width=True)
        
        # Enhanced legend with insights
        perfect_days = len(calendar_df[calendar_df['intensity'] == 4])
        good_days = len(calendar_df[calendar_df['intensity'] >= 2])
        
        cal_col1, cal_col2, cal_col3 = st.columns(3)
        cal_col1.metric("🔥 Goal Achieved Days", perfect_days)
        cal_col2.metric("💪 Productive Days", good_days)
        cal_col3.metric("📈 Success Rate", f"{(good_days/42*100):.0f}%")
        
        st.markdown("""
        **Legend:** 🔥 Goal Crushed (Target met) | 💪 Strong (50+ min) | ⚡ Good Start (25+ min) | 👶 Baby Steps (<25 min) | ❌ Rest Day (0 min)
        """)
    
    # 8. Weekend vs Weekday Analysis
    st.subheader("🏖️ Weekend vs Weekday Performance")
    
    df_work_copy = df_work.copy()
    df_work_copy['is_weekend'] = df_work_copy['date'].dt.dayofweek >= 5
    
    weekend_analysis = df_work_copy.groupby('is_weekend').agg({
        'duration': ['sum', 'mean', 'count']
    }).round(2)
    weekend_analysis.columns = ['Total Minutes', 'Avg Minutes', 'Sessions']
    weekend_analysis.index = ['Weekdays', 'Weekends']
    
    ww_col1, ww_col2 = st.columns(2)
    
    with ww_col1:
        ww_fig = px.bar(weekend_analysis.reset_index(), x='index', y='Total Minutes',
                       title="Weekday vs Weekend Total",
                       color='Total Minutes', color_continuous_scale='Blues')
        st.plotly_chart(ww_fig, use_container_width=True)
    
    with ww_col2:
        avg_fig = px.bar(weekend_analysis.reset_index(), x='index', y='Avg Minutes',
                        title="Average Daily Performance",
                        color='Avg Minutes', color_continuous_scale='Greens')
        st.plotly_chart(avg_fig, use_container_width=True)
    
    # Weekend insights
    weekend_ratio = weekend_analysis.loc['Weekends', 'Avg Minutes'] / weekend_analysis.loc['Weekdays', 'Avg Minutes'] if weekend_analysis.loc['Weekdays', 'Avg Minutes'] > 0 else 0
    
    if weekend_ratio > 0.8:
        st.success("🏆 **Weekend Warrior:** You maintain great productivity on weekends!")
    elif weekend_ratio > 0.5:
        st.info("⚖️ **Balanced:** Good weekend vs weekday balance.")
    else:
        st.warning("🔄 **Weekend Opportunity:** Consider light weekend sessions to maintain momentum.")
    
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

