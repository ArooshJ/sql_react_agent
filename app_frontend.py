"""
Streamlit Frontend for SQL ReAct Agent
Optimized for performance and UX.
"""

import streamlit as st
import requests
from datetime import datetime

# API configuration
API_URL = "http://localhost:5000"

# Page config
st.set_page_config(
    page_title="SQL ReAct Agent",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'agent_initialized' not in st.session_state:
    st.session_state.agent_initialized = False
if 'api_healthy' not in st.session_state:
    st.session_state.api_healthy = False
if 'health_checked' not in st.session_state:
    st.session_state.health_checked = False

def check_health():
    """Check API health only once per session/reset."""
    if st.session_state.health_checked:
        return st.session_state.api_healthy
    
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        if health.get("status") == "healthy":
            st.session_state.api_healthy = True
        else:
            st.session_state.api_healthy = False
    except:
        st.session_state.api_healthy = False
    
    st.session_state.health_checked = True
    return st.session_state.api_healthy

def init_agent():
    """Initialize agent via API."""
    try:
        response = requests.post(
            f"{API_URL}/init", 
            json={"db_path": "database/company.db"},
            timeout=10
        )
        if response.status_code == 200:
            st.session_state.agent_initialized = True
            return True
        return False
    except:
        return False

def run_query(query):
    """Run query via API."""
    try:
        response = requests.post(
            f"{API_URL}/query", 
            json={"query": query},
            timeout=120
        )
        if response.status_code == 200:
            return response.json()
        return {"error": response.json().get('error', 'Unknown error'), "steps": []}
    except Exception as e:
        return {"error": str(e), "steps": []}

# --- MAIN LOGIC ---

# 1. Perform Health Check & Auto-Init (Once)
is_healthy = check_health()

if is_healthy and not st.session_state.agent_initialized:
    # Auto-init silently or with a quick spinner
    init_agent()

# --- UI LAYOUT ---

st.title("🤖 SQL ReAct Agent")

# Sidebar
with st.sidebar:
    st.header("Status")
    if st.session_state.api_healthy:
        st.success("🟢 API Online")
    else:
        st.error("🔴 API Offline")
        st.info(f"Ensure backend is running at {API_URL}")
    
    if st.session_state.agent_initialized:
        st.success("🟢 Agent Ready")
    else:
        st.warning("🔴 Agent Not Ready")
        if st.button("Retry Connection"):
            st.session_state.health_checked = False # Force recheck
            st.rerun()

    st.divider()
    st.markdown("### Recent History")
    for item in reversed(st.session_state.query_history[-5:]): # Last 5
        with st.expander(f"Q: {item['query'][:25]}..."):
            st.caption(f"A: {item['result'].get('final_answer', '...')}")

# Main Area
user_query = st.text_area("Ask a question about the database:", height=100, placeholder="e.g., How many employees are in Engineering?")

col_btn, col_opt = st.columns([1, 3])
with col_btn:
    run_clicked = st.button("🚀 Run Query", type="primary", disabled=not st.session_state.agent_initialized)
with col_opt:
    show_details = st.checkbox("Show Reasoning (ReAct Loop)", value=True)

if run_clicked and user_query:
    with st.spinner("Thinking..."):
        result = run_query(user_query)
        
        if "error" in result:
            st.error(f"❌ Error: {result['error']}")
        else:
            # Save to history
            st.session_state.query_history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "query": user_query,
                "result": result
            })
            
            # Display Result
            st.markdown("### ✅ Answer")
            st.success(result.get("final_answer", "No answer provided."))
            
            # Display Reasoning if toggled
            if show_details:
                st.markdown("#### 🕵️ Reasoning Process")
                for step in result.get("steps", []):
                    if step["type"] == "REACT_CYCLE":
                        with st.expander(f"Step {step['iteration']}: {step['action']}", expanded=True):
                            if step.get("thought"):
                                st.info(f"🧠 **Thought:** {step['thought']}")
                            st.code(f"🔧 Action: {step['action']}({step['action_params']})", language="json")
                            st.text(f"👁️ Observation: {step['observation']}")
