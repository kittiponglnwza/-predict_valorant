"""
ui/styles.py — Global CSS styles for Football AI Nexus Engine
"""
import streamlit as st


def inject_global_css():
    """Inject the global CSS into the Streamlit app."""
    st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    .stApp { 
        background-color: #0B0F19; 
        color: #E2E8F0;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #151B2B, #0B0F19);
        border: 1px solid #2A3441;
        padding: 1.25rem;
        border-radius: 16px;
        border-left: 4px solid #00B0FF;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0, 176, 255, 0.15);
        border-color: #00B0FF;
    }
    
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: #111622;
        border-radius: 12px;
        padding: 6px;
        gap: 8px;
        border: 1px solid #2A3441;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: transparent;
        border-radius: 8px;
        color: #94A3B8;
        padding: 0 20px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1F2937 !important;
        color: #00B0FF !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #00B0FF 0%, #0081CB 100%);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 176, 255, 0.2);
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 16px rgba(0, 176, 255, 0.4);
        background: linear-gradient(90deg, #1AD6FF 0%, #00B0FF 100%);
    }
    
    [data-testid="stSidebar"] {
        background-color: #0E131F !important;
        border-right: 1px solid #1F2937;
    }
    
    h1, h2, h3 {
        color: #F8FAFC;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #00B0FF, #00E676);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)