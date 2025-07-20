# Time_Tracker

A web-based productivity app for tracking Pomodoro sessions, breaks, and daily notes, with multi-user support and analytics.

## Features

- **Multi-user support:** Add and select users, each with their own data and progress.
- **Pomodoro timer:** Start 25-minute work sessions and 5-minute breaks.
- **Category & Task tracking:** Assign categories and tasks to each session.
- **Notes:** Save and view daily notes.
- **Analytics:** Visualize productivity, streaks, and time spent per task/category.
- **MongoDB backend:** All data is securely stored and managed in MongoDB.
- **Custom categories:** Add your own categories for better organization.
- **Sound and visual alerts:** Get notified when sessions end.

## Usage Instructions

1. **Setup:**
   - Install requirements:  
     `pip install streamlit pymongo pandas plotly pytz`
   - Set up a MongoDB database and add your connection string to Streamlit secrets as `mongo_uri`.

2. **Run the app:**
   - In the project directory, run:  
     `streamlit run time_tracker.py`

3. **Using the app:**
   - **User selection:** Use the sidebar to select or add a user.
   - **Pomodoro Tracker:**  
     - Select a category or add a new one.
     - Enter a task name.
     - Start a work session or break.
   - **Notes:**  
     - Save daily notes in "Notes Saver".
     - View notes by date range in "Notes Viewer".
   - **Analytics:**  
     - View daily/overall productivity, streaks, and breakdowns by category/task.

4. **Data:**  
   - All Pomodoro sessions and notes are linked to the selected user.
   - Existing data is automatically assigned to the default user `prashanth` on first run.

## Screenshots

*(Add screenshots here if desired)*

## License

MIT License