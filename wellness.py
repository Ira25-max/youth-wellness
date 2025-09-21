import os
import json
import datetime
import streamlit as st
import plotly.express as px
import google.generativeai as genai
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from api import GENAI_API_KEY
import html
import re
import json
#from datetime import datetime, date



# =========================
# 🔑 CONFIGURATION
# =========================
st.set_page_config(page_title="MindHaven AI", page_icon="🧠", layout="centered")

if not GENAI_API_KEY:
    st.error("⚠️ Please set GENAI_API_KEY in api.py.")
else:
    genai.configure(api_key=GENAI_API_KEY)

try:
    model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    st.error(f"Model load failed: {e}")
    model = None

JOURNAL_FILE = "journal.json"
COMMUNITY_FILE = "community.json"

# =========================
# 🤖 SAFE AI WRAPPER
# =========================
def safe_ai(prompt: str) -> str:
    if not model:
        return "⚠️ AI not available."
    try:
        response = model.generate_content(prompt)
        return response.text if hasattr(response, "text") else str(response)
    except Exception as e:
        return f"⚠️ AI unavailable: {e}"

# =========================
# 📓 JOURNAL HELPERS
# =========================
def save_journal(entry: str, mood: str):
    data = load_journal()
    today = datetime.date.today().strftime("%Y-%m-%d")
    data.append({"date": today, "entry": entry, "mood": mood})
    with open(JOURNAL_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_journal():
    if not os.path.exists(JOURNAL_FILE):
        return []
    with open(JOURNAL_FILE, "r") as f:
        data = json.load(f)
    new_data = []
    for item in data:
        if isinstance(item, list) or isinstance(item, tuple):
            new_data.append({"date": item[0], "entry": item[1], "mood": ""})
        else:
            new_data.append(item)
    return new_data

# =========================
# 💬 CHAT HELPERS
# =========================
CHAT_FILE = "chat_history.json"

def save_chat():
    """Save the current chat history to a JSON file."""
    if "chat_history" not in st.session_state:
        return
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.chat_history, f, indent=2, ensure_ascii=False)

def load_chat():
    """Load saved chat history if it exists."""
    if not os.path.exists(CHAT_FILE):
        return []
    with open(CHAT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# =========================
# 📊 MOOD ANALYSIS
# =========================
def analyze_mood(journal):
    if not journal:
        st.info("No journal entries yet.")
        return

    df = pd.DataFrame(journal)
    df["date"] = pd.to_datetime(df["date"])

    moods = ["happy", "sad", "angry", "anxious", "excited"]
    selected_moods = st.multiselect("Select moods to display", moods, default=moods)

    for mood in moods:
        df[mood] = df["mood"].str.lower().str.contains(mood).astype(int)

    df_grouped = df.groupby("date")[selected_moods].sum().reset_index()

    st.markdown("### Mood Changes Over Time")
    fig_line = px.line(df_grouped, x="date", y=selected_moods, title="Mood Trends")
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("### Mood Distribution")
    mood_counts = {m: df[m].sum() for m in selected_moods if df[m].sum() > 0}
    if mood_counts:
        fig_pie = px.pie(names=list(mood_counts.keys()), values=list(mood_counts.values()))
        st.plotly_chart(fig_pie, use_container_width=True)

# =========================
# 📝 WEEKLY SUMMARY
# =========================
def weekly_summary(journal):
    today = datetime.date.today()
    now = datetime.datetime.now()
    week_ago = today - datetime.timedelta(days=7)
    week_entries = [
        j for j in journal
        if datetime.datetime.strptime(j['date'], "%Y-%m-%d").date() >= week_ago
    ]

    if not week_entries:
        st.info("No entries in the past week.")
        return

    df = pd.DataFrame(week_entries)
    df["date"] = pd.to_datetime(df["date"])

    moods = ["happy", "sad", "angry", "anxious", "excited"]
    mood_counts = {m: df["mood"].str.lower().str.contains(m).sum() for m in moods}

    dominant = max(mood_counts, key=mood_counts.get)
    st.markdown(f"**Most frequent mood this week:** {dominant} ({mood_counts[dominant]} times)")

    for mood in moods:
        df[mood] = df["mood"].str.lower().str.contains(mood).astype(int)
    df_grouped = df.groupby("date")[moods].sum().reset_index()

    fig_line = px.line(df_grouped, x="date", y=moods, title="Mood Trends (7 days)")
    st.plotly_chart(fig_line, use_container_width=True)

# =========================
# 🌟 AI-ENHANCED FEATURES
# =========================
def summarize_journal(entries):
    text = "\n".join([e["entry"] for e in entries])
    return safe_ai(f"Summarize the emotional patterns in this journal:\n{text}")

def recommend_action(entries):
    text = "\n".join([e["entry"] for e in entries[-5:]])
    return safe_ai(f"Based on these journal entries, suggest 3 short wellness tips:\n{text}")

def wordcloud_viz(entries):
    all_text = " ".join([e["entry"] for e in entries])
    wc = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# =========================
# 🌙 DARK MODE
# =========================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode)
st.session_state.dark_mode = dark_mode
if dark_mode:
    st.markdown(
        "<style>body{background-color:#1E1E1E;color:#F0F0F0;}</style>",
        unsafe_allow_html=True,
    )


def clean_response(msg: str) -> str:
    # remove all literal </div> occurrences
    return msg.replace("</div>", "").replace("<div>", "")

# =========================
# 🎨 CHAT DISPLAY WITH HIGHLIGHT
# =========================
def display_chat():
    for speaker, msg, timestamp in st.session_state.chat_history:
        # Clean/sanitize message (escape raw HTML tags so they don’t render)
        safe_msg = html.escape(msg)

        if speaker == "You":
            st.markdown(
                f"""
                <div style="
                    background-color: #fff3cd;
                    padding: 12px;
                    border-radius: 10px;
                    margin: 6px 0;
                    border-left: 5px solid #ffca2c;
                ">
                    <b>🧑 You [{timestamp}]</b><br>{safe_msg}
                    
                """,
                unsafe_allow_html=True
            )
        else:  # AI
            st.markdown(
                f"""
                <div style="
                    background-color: #e2f0d9;
                    padding: 12px;
                    border-radius: 10px;
                    margin: 6px 0;
                    border-left: 5px solid #28a745;
                ">
                    <b>🤖 {speaker} [{timestamp}]</b><br>{safe_msg}
                
                """,
                unsafe_allow_html=True
            )

# =========================
# 🌐 STREAMLIT APP
# =========================
st.title("🧠 MindHaven AI – Youth Mental Wellness Companion")

menu = st.sidebar.radio("Navigate", ["💬 Chat", "📓 Journal", "📊 Mood Trends", "📝 Weekly Summary", "🌍 Community"])

# =========================
# 💬 CHAT
# =========================



if menu == "💬 Chat":
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat()  # load previous chat if available

    # Display chat with highlight
    display_chat()

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("How are you feeling today?", height=80)
        submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        ts = datetime.datetime.now().strftime("%H:%M")
        ai_response = safe_ai(user_input)
        st.session_state.chat_history.append(("You", user_input, ts))
        st.session_state.chat_history.append(("MindHaven AI", ai_response, ts))
        save_chat()  # auto-save after every new message
        st.rerun()

    # Manual save button
    if st.button("💾 Save Chat to Journal", key="save_chat_btn"):
        save_chat()
        st.success("Chat saved to chat_history.json")



# =========================
# 📓 JOURNAL
# =========================
elif menu == "📓 Journal":
    entry = st.text_area("Write your thoughts...")
    mood = st.selectbox("Select mood", ["", "happy", "sad", "angry", "anxious", "excited"])
    if st.button("Save Entry"):
        if entry.strip():
            save_journal(entry, mood)
            st.session_state.last_entry, st.session_state.last_mood = entry, mood
            st.success("Entry saved ✅")
        else:
            st.warning("Write something first!")

    st.subheader("Previous Entries")
    journal = load_journal()
    for item in reversed(journal):
        st.markdown(f"**{item['date']} ({item['mood']})** – {item['entry']}")

    if journal:
        st.subheader("📊 Word Cloud of Entries")
        wordcloud_viz(journal)

# =========================
# 📊 MOOD TRENDS
# =========================
elif menu == "📊 Mood Trends":
    analyze_mood(load_journal())

# =========================
# 📝 WEEKLY SUMMARY
# =========================
elif menu == "📝 Weekly Summary":
    journal = load_journal()
    weekly_summary(journal)
    if journal:
        st.subheader("AI Reflection")
        st.write(summarize_journal(journal[-7:]))
        st.subheader("💡 Personalized Tips")
        st.write(recommend_action(journal))

# =========================
# 🌍 COMMUNITY
# =========================
elif menu == "🌍 Community":
    if st.checkbox("Share last entry anonymously"):
        if "last_entry" in st.session_state:
            entry = {"date": datetime.date.today().isoformat(),
                     "entry": st.session_state.last_entry,
                     "mood": st.session_state.last_mood}
            community = []
            if os.path.exists(COMMUNITY_FILE):
                with open(COMMUNITY_FILE, "r") as f:
                    community = json.load(f)
            community.append(entry)
            with open(COMMUNITY_FILE, "w") as f:
                json.dump(community, f, indent=2)
            st.success("Shared to community 🌍")
        else:
            st.warning("No recent entry to share.")

    if st.button("View Community Feed"):
        if os.path.exists(COMMUNITY_FILE):
            with open(COMMUNITY_FILE, "r") as f:
                feed = json.load(f)
            for post in reversed(feed):
                st.info(f"📝 {post['entry']}  \n🙂 Mood: {post['mood']} ({post['date']})")
        else:
            st.info("No posts yet.")
