import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import requests
from streamlit_lottie import st_lottie

# ---------------------------
# Load and Prepare Dataset
# ---------------------------
@st.cache_data
def load_data():
    data = pd.read_csv('/Users/pradeepkumar/Desktop/Project1/spam.csv')
    data.drop_duplicates(inplace=True)
    data["Category"] = data["Category"].replace(["ham", "spam"], ["Not Spam", "Spam"])
    return data

data = load_data()

X = data["Message"]
y = data["Category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# TF-IDF Vectorization
# ---------------------------
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---------------------------
# Train the model
# ---------------------------
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ---------------------------
# Prediction Function
# ---------------------------
def predict(message):
    transformed = vectorizer.transform([message])
    prediction = model.predict(transformed)[0]
    confidence = model.predict_proba(transformed).max()
    return prediction, round(confidence * 100, 2)

# ---------------------------
# Load Lottie Animations
# ---------------------------
@st.cache_data
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None

spam_lottie = load_lottie_url("https://assets6.lottiefiles.com/packages/lf20_er7lgjhc.json")
clean_lottie = load_lottie_url("https://assets6.lottiefiles.com/packages/lf20_2cwDXD.json")
welcome_lottie = load_lottie_url("https://assets6.lottiefiles.com/packages/lf20_5tkzkblw.json")

# ---------------------------
# Streamlit App Layout
# ---------------------------
st.set_page_config(
    page_title="Spam Detector", 
    page_icon="📩", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 10px !important;
        padding: 12px !important;
    }
    .stButton>button {
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: bold !important;
        background: linear-gradient(45deg, #4CAF50, #2E7D32) !important;
        color: white !important;
        border: none !important;
        width: 100%;
        margin: 5px 0;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #2E7D32, #1B5E20) !important;
    }
    .reset-button>button {
        background: linear-gradient(45deg, #f44336, #c62828) !important;
    }
    .reset-button>button:hover {
        background: linear-gradient(45deg, #c62828, #b71c1c) !important;
    }
    .result-box {
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .spam-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .ham-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Smart Spam Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; margin-bottom: 30px;'>Detect spam email messages with AIML</p>", unsafe_allow_html=True)

# Welcome Animation
if welcome_lottie:
    st_lottie(welcome_lottie, speed=1, height=200, key="welcome")

# Initialize session state for message
if 'message_input' not in st.session_state:
    st.session_state.message_input = ""

# Main Content
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Enter Your Message")
        user_input = st.text_area(
            "", 
            height=150, 
            placeholder="Paste your message here...",
            label_visibility="collapsed",
            value=st.session_state.message_input,
            key="message_display"
        )
    
    with col2:
        st.markdown("### Quick Stats")
        st.metric("Accuracy", f"{round(model.score(X_test_tfidf, y_test) * 100, 2)}%")
        st.metric("Total Samples", len(data))
        st.metric("Spam Ratio", f"{round(data['Category'].value_counts()['Spam']/len(data)*100,1)}%")

# Action Buttons
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("Analyze Message", key="analyze"):
        if not st.session_state.message_input.strip():
            st.warning("Please enter a message to analyze")
        else:
            with st.spinner('Analyzing message...'):
                prediction, confidence = predict(st.session_state.message_input)
                
                if prediction == "Spam":
                    st.markdown(f"""
                    <div class="result-box spam-box">
                        <h3 style='color: #f44336;'>Spam Detected!</h3>
                        <p>Confidence: <strong>{confidence}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    if spam_lottie:
                        st_lottie(spam_lottie, height=200, key="spam")
                else:
                    st.markdown(f"""
                    <div class="result-box ham-box">
                        <h3 style='color: #4CAF50;'>Not Spam</h3>
                        <p>Confidence: <strong>{confidence}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    if clean_lottie:
                        st_lottie(clean_lottie, height=200, key="ham")

with col3:
    st.markdown("<div class='reset-button'>", unsafe_allow_html=True)
    if st.button("Clear Message", key="reset"):
        st.session_state.message_input = ""
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Sample Messages
st.markdown("---")
st.markdown("### Try Sample Messages")

col1, col2 = st.columns(2)
with col1:
    if st.button("Test Spam Message", key="test_spam"):
        st.session_state.message_input = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!"
        st.rerun()

with col2:
    if st.button("Test Normal Message", key="test_ham"):
        st.session_state.message_input = "Hey, just checking if we're still meeting for lunch tomorrow at 1pm?"
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>Built with ❤ by Pradeep Naina & Abhishek</p>
    <p>Mentor: Sourav Sir</p>
</div>
""", unsafe_allow_html=True)