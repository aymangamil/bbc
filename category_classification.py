import streamlit as st
import pickle
import re
import string
import nltk
import pandas as pd
import numpy as np
import plotly.express as px
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# --- NLTK Setup ---
nltk_resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
for resource in nltk_resources:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

# --- Load Model and Vectorizer ---
@st.cache_resource
def load_assets():
    try:
        with open("model_Log.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vec.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Error: Required model files not found. Please ensure 'model_Log.pkl' and 'vec.pkl' are available.")
        st.stop()

model, vectorizer = load_assets()

# --- Text Preprocessing ---
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    words = nltk.word_tokenize(text)
    words = [w for w in words if w not in stop_words and len(w) > 2]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# --- Premium CSS Styling ---
st.markdown("""
<style>
    /* Base Theme */
    :root {
        --primary: #4361ee;
        --secondary: #3a0ca3;
        --accent: #f72585;
        --dark: #212529;
        --darker: #1a1a2e;
        --light: #f8f9fa;
        --success: #4cc9f0;
        --warning: #f8961e;
        --danger: #ef233c;
        --card-bg: rgba(30, 30, 60, 0.7);
        --glass: rgba(255, 255, 255, 0.05);
        --transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    
    /* Main Layout */
    .stApp {
        background: linear-gradient(135deg, var(--darker) 0%, var(--dark) 100%);
        background-attachment: fixed;
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        color: var(--light);
    }
    
    /* Header */
    .header {
        background: linear-gradient(90deg, var(--secondary) 0%, var(--primary) 100%);
        padding: 2rem 1rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .header::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Title */
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        position: relative;
        text-align: center;
        background: linear-gradient(90deg, #fff 0%, var(--accent) 50%, var(--success) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    /* Input Area */
    .stTextArea textarea {
        background: var(--card-bg) !important;
        border: 1px solid var(--glass) !important;
        color: var(--light) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        font-size: 1rem !important;
        transition: var(--transition) !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    .stTextArea textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(67, 97, 238, 0.3) !important;
        outline: none !important;
    }
    
    /* Button */
    .stButton button {
        background: linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: var(--transition) !important;
        box-shadow: 0 4px 15px rgba(67, 97, 238, 0.3) !important;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 20px rgba(67, 97, 238, 0.4) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--card-bg) !important;
        border-radius: 10px !important;
        padding: 0.5rem 1.5rem !important;
        margin: 0 !important;
        transition: var(--transition) !important;
        border: 1px solid var(--glass) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 5px 15px rgba(67, 97, 238, 0.3) !important;
    }
    
    /* Cards */
    .card {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid var(--glass);
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        transition: var(--transition);
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
    }
    
    /* Category Badges */
    .badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .business { background: linear-gradient(90deg, #f8961e 0%, #f9c74f 100%); }
    .entertainment { background: linear-gradient(90deg, #b5179e 0%, #f72585 100%); }
    .politics { background: linear-gradient(90deg, #ef233c 0%, #d90429 100%); }
    .sport { background: linear-gradient(90deg, #4cc9f0 0%, #4895ef 100%); }
    .tech { background: linear-gradient(90deg, #2ec4b6 0%, #2bbd7e 100%); }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .title {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Category Data (Corrected for Plotly) ---
categories = {
    'business': {
        'label': 'Business',
        'icon': '�',
        'css_color': 'linear-gradient(90deg, #f8961e 0%, #f9c74f 100%)',
        'plotly_color': '#f8961e',
        'description': 'Corporate, financial, and economic news'
    },
    'entertainment': {
        'label': 'Entertainment',
        'icon': '🎭',
        'css_color': 'linear-gradient(90deg, #b5179e 0%, #f72585 100%)',
        'plotly_color': '#b5179e',
        'description': 'Celebrity, movies, music, and arts'
    },
    'politics': {
        'label': 'Politics',
        'icon': '🏛️',
        'css_color': 'linear-gradient(90deg, #ef233c 0%, #d90429 100%)',
        'plotly_color': '#ef233c',
        'description': 'Government, elections, and policy'
    },
    'sport': {
        'label': 'Sports',
        'icon': '⚽',
        'css_color': 'linear-gradient(90deg, #4cc9f0 0%, #4895ef 100%)',
        'plotly_color': '#4cc9f0',
        'description': 'Athletics, competitions, and matches'
    },
    'tech': {
        'label': 'Technology',
        'icon': '💻',
        'css_color': 'linear-gradient(90deg, #2ec4b6 0%, #2bbd7e 100%)',
        'plotly_color': '#2ec4b6',
        'description': 'Innovations, gadgets, and digital trends'
    }
}

# --- App Header ---
st.markdown("""
<div class="header">
    <h1 class="title">BBC News Category Classifier</h1>
</div>
""", unsafe_allow_html=True)

# --- Main Content ---
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 📝 Enter News Text")
    user_input = st.text_area("", height=200, placeholder="Paste news text here for classification...", label_visibility="collapsed")

    if st.button("**Predict Category**"):
        if not user_input.strip():
            st.warning("Please enter text to classify")
        else:
            with st.spinner("Analyzing text..."):
                cleaned_text = clean_text(user_input)
                vectorized = vectorizer.transform([cleaned_text])
                prediction = model.predict(vectorized)[0]
                probabilities = model.predict_proba(vectorized)[0]
                
                # Store results in session state
                st.session_state.prediction = prediction
                st.session_state.probabilities = probabilities

with col2:
    st.markdown("### ℹ️ About This App")
    st.markdown("""
    <div class="card">
        <p>This application uses machine learning to classify news articles into BBC's main categories:</p>
        <ul style="margin-top: 0.5rem;">
            <li>Business 💼</li>
            <li>Entertainment 🎭</li>
            <li>Politics 🏛️</li>
            <li>Sports ⚽</li>
            <li>Technology 💻</li>
        </ul>
        <p style="margin-top: 1rem;">Enter your text and click predict to get results.</p>
    </div>
    """, unsafe_allow_html=True)

# --- Results Display ---
if 'prediction' in st.session_state:
    prediction = st.session_state.prediction
    probabilities = st.session_state.probabilities
    
    result_tab, confidence_tab = st.tabs(["🎯 Prediction Result", "📊 Confidence Analysis"])
    
    with result_tab:
        category = categories[prediction]
        st.markdown(f"""
        <div class="card">
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
                <div style="font-size: 2.5rem;">{category['icon']}</div>
                <div>
                    <h3 style="margin: 0; color: {category['plotly_color']};">{category['label']}</h3>
                    <p style="margin: 0; opacity: 0.8;">Predicted Category</p>
                </div>
            </div>
            <div style="background: {category['css_color']}; height: 6px; border-radius: 3px; margin-bottom: 1rem;"></div>
            <p style="font-size: 1.1rem;">The input text belongs to <strong>{category['label']}</strong> category with high confidence.</p>
            <p style="font-size: 0.9rem; opacity: 0.8; margin-top: 0.5rem;">{category['description']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with confidence_tab:
        # Create dataframe for visualization
        prob_df = pd.DataFrame({
            'Category': [categories[cat]['label'] for cat in categories],
            'Confidence': probabilities,
            'PlotlyColor': [categories[cat]['plotly_color'] for cat in categories], # Use Plotly-compatible colors
            'Icon': [categories[cat]['icon'] for cat in categories]
        })
        
        # Sort by confidence
        prob_df = prob_df.sort_values('Confidence', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            prob_df,
            x='Category',
            y='Confidence',
            color='Category',
            color_discrete_map={row['Category']: row['PlotlyColor'] for _, row in prob_df.iterrows()},
            text='Icon',
            labels={'Confidence': 'Confidence Score', 'Category': ''},
            height=400
        )
        
        fig.update_traces(
            textposition='outside',
            marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>Confidence: %{y:.1%}<extra></extra>"
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis=dict(tickangle=-45),
            yaxis=dict(tickformat=".0%"),
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=20))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show confidence table
        st.markdown("### 📈 Detailed Confidence Scores")
        prob_df['Confidence'] = prob_df['Confidence'].apply(lambda x: f"{x*100:.1f}%")
        st.dataframe(
            prob_df[['Icon', 'Category', 'Confidence']].reset_index(drop=True),
            hide_index=True,
            use_container_width=True
        )

