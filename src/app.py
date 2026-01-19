import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# --- 1. CONFIGURATION & SIDEBAR ---
HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
# Using Llama 3.3 for the smartest schema mapping
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct:groq"

st.set_page_config(layout="wide", page_title="Universal CSV Analyzer")

with st.sidebar:
    st.header("🔑 API Setup")
    api_token = st.text_input("Hugging Face Token", type="password")
    st.info("This app uses AI to automatically map your headers to the correct analysis.")

# --- 2. DYNAMIC MAPPING LOGIC ---
def get_column_mapping(headers, api_token):
    """Asks the LLM to categorize columns based on their names."""
    prompt = f"""
    Given these CSV headers: {headers}
    Map exactly ONE header to each of these categories if applicable:
    - RATING: A numeric score or attractiveness rating.
    - MOTIVATOR: Factors for applying/motivation.
    - METRIC: A percentage or count (like completion rates).
    - CATEGORY: A group or question label.
    
    Return ONLY a JSON object like: {{"RATING": "col_name", "MOTIVATOR": "col_name"...}}
    """
    headers_req = {"Authorization": f"Bearer {api_token}"}
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": { "type": "json_object" } # Ensure valid JSON
    }
    try:
        res = requests.post(HF_ROUTER_URL, headers=headers_req, json=payload)
        return res.json()['choices'][0]['message']['content']
    except:
        return None

# --- 3. MAIN APP ---
st.title("🚀 Universal Survey Analytics")
uploaded = st.file_uploader("Upload ANY CSV file", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    st.write(f"📂 Loaded: **{uploaded.name}** ({len(df)} rows)")
    
    # Automatically Detect Data Types
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()

    with st.expander("🛠️ Step 1: Intelligent Header Mapping"):
        if st.button("Auto-Detect Columns with AI"):
            if not api_token:
                st.warning("Please enter your token in the sidebar.")
            else:
                mapping = get_column_mapping(df.columns.tolist(), api_token)
                st.session_state.mapping = mapping
                st.json(mapping)
        
        # Manual Override/Fallback
        target_col = st.selectbox("Select Primary Metric to Visualize", num_cols if num_cols else df.columns)
        label_col = st.selectbox("Select Category/Label Column", cat_cols if cat_cols else df.columns)

    # --- 4. UNIVERSAL VISUALS ---
    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📊 Distribution")
        # Works for ANY numeric column
        if not df[target_col].empty:
            fig_dist = px.histogram(df, x=target_col, title=f"Distribution of {target_col}", 
                                    color_discrete_sequence=['#00CC96'])
            st.plotly_chart(fig_dist, use_container_width=True)

    with c2:
        st.subheader("🏆 Top Factors")
        # Works for ANY categorical data
        top_data = df.groupby(label_col)[target_col].mean().sort_values(ascending=False).head(10).reset_index()
        fig_bar = px.bar(top_data, x=label_col, y=target_col, title=f"Average {target_col} by {label_col}")
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- 5. AI EXECUTIVE SUMMARY ---
    st.divider()
    if st.button("✨ Generate Summary for THIS File"):
        sample_data = df.head(5).to_string()
        prompt = f"Analyze this specific dataset: {uploaded.name}. Data sample: {sample_data}. Summarize the top 3 insights."
        
        # Use same payload structure as before
        payload = {"model": MODEL_ID, "messages": [{"role": "user", "content": prompt}]}
        with st.spinner("Analyzing..."):
            res = requests.post(HF_ROUTER_URL, headers={"Authorization": f"Bearer {api_token}"}, json=payload)
            if res.status_code == 200:
                st.info(res.json()['choices'][0]['message']['content'])