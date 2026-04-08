import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- PAGE SETUP ---
st.set_page_config(
    page_title="AI Sales Predictor",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# --- GLOBAL CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] .stSlider > div > div > div > div {
        background: #6c63ff !important;
    }

    /* Header banner */
    .hero-banner {
        background: linear-gradient(135deg, #6c63ff 0%, #3f3d8f 50%, #0f0c29 100%);
        padding: 40px 40px 30px 40px;
        border-radius: 20px;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(108, 99, 255, 0.35);
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 200px; height: 200px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        bottom: -40px; left: -40px;
        width: 150px; height: 150px;
        background: rgba(255,255,255,0.03);
        border-radius: 50%;
    }
    .hero-title {
        color: #ffffff;
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.7);
        font-size: 1.05rem;
        margin-top: 8px;
        font-weight: 400;
    }

    /* Glass cards */
    .glass-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 20px;
        transition: box-shadow 0.3s ease;
    }
    .glass-card:hover {
        box-shadow: 0 8px 40px rgba(108,99,255,0.25);
    }

    /* Section headers */
    .section-header {
        color: #a78bfa;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .section-title {
        color: #ffffff;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 20px;
    }

    /* Prediction result card */
    .result-card {
        background: linear-gradient(135deg, #6c63ff 0%, #a855f7 100%);
        padding: 36px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 20px 60px rgba(108, 99, 255, 0.45);
        margin: 10px 0;
        animation: glow-pulse 3s ease-in-out infinite;
    }
    @keyframes glow-pulse {
        0%, 100% { box-shadow: 0 20px 60px rgba(108,99,255,0.45); }
        50%       { box-shadow: 0 20px 80px rgba(108,99,255,0.70); }
    }
    .result-value {
        font-size: 4.5rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1;
        letter-spacing: -2px;
    }
    .result-unit {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.75);
        font-weight: 500;
        margin-top: 6px;
    }
    .result-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 14px;
    }

    /* KPI metric cards */
    .kpi-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        margin-bottom: 14px;
    }
    .kpi-label {
        color: rgba(255,255,255,0.55);
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .kpi-value {
        color: #ffffff;
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 6px;
    }
    .kpi-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 30px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-top: 6px;
        background: rgba(108,99,255,0.35);
        color: #c4b5fd;
    }

    /* Tag display */
    .tag {
        display: inline-block;
        background: rgba(108,99,255,0.25);
        color: #c4b5fd;
        border: 1px solid rgba(108,99,255,0.4);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 3px;
    }

    /* Button styling override */
    .stButton > button {
        background: linear-gradient(135deg, #6c63ff, #a855f7) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 32px !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.5px !important;
        width: 100% !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        box-shadow: 0 8px 30px rgba(108,99,255,0.4) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 40px rgba(108,99,255,0.6) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.08) !important; }

    /* Streamlit element label colors */
    label, .stSelectbox label, .stSlider label, .stNumberInput label {
        color: rgba(255,255,255,0.75) !important;
        font-weight: 500 !important;
    }

    /* Input boxes */
    input[type="number"], .stSelectbox > div > div {
        background: rgba(255,255,255,0.08) !important;
        border-color: rgba(255,255,255,0.15) !important;
        color: #fff !important;
        border-radius: 10px !important;
    }

    /* Info and error boxes */
    .stAlert { border-radius: 12px !important; }

    /* Footer */
    .footer-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 22px 28px;
        color: rgba(255,255,255,0.55);
        font-size: 0.88rem;
        line-height: 1.7;
        margin-top: 10px;
    }
    .footer-card strong { color: #a78bfa; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); }
    ::-webkit-scrollbar-thumb { background: #6c63ff; border-radius: 3px; }
    </style>
""", unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('sales_model.pkl')
        features = joblib.load('feature_names.pkl')
        return model, features, None
    except Exception as e:
        return None, None, str(e)

model, model_features, load_error = load_assets()

if model is None:
    st.error(f"⚠️ **Could not load model assets:** {load_error}")
    st.info("Ensure `sales_model.pkl` and `feature_names.pkl` exist in the same directory.")
    st.stop()

# Extract unique categories and cities from features
categories = sorted([f.replace("category_", "") for f in model_features if f.startswith("category_")])
cities     = sorted([f.replace("city_", "")     for f in model_features if f.startswith("city_")])

# --- HERO BANNER ---
st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">📊 AI Sales Forecasting Tool</div>
        <div class="hero-subtitle">Powered by Random Forest · Predict monthly unit sales with confidence</div>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <div style="font-size:2rem;">🕹️</div>
        <div style="color:#a78bfa; font-weight:700; font-size:1.1rem; letter-spacing:0.5px;">Control Panel</div>
        <div style="color:rgba(255,255,255,0.4); font-size:0.8rem; margin-top:4px;">Adjust parameters below</div>
    </div>
    <hr style="border-color:rgba(255,255,255,0.08); margin-bottom:20px;">
""", unsafe_allow_html=True)

st.sidebar.markdown("**💰 Pricing**")
price    = st.sidebar.number_input("Unit Price ($)", min_value=1.0, max_value=1000.0, value=50.0, step=1.0)
discount = st.sidebar.slider("Discount (%)", 0.0, 80.0, 10.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.markdown("**⭐ Product Quality**")
rating     = st.sidebar.slider("Customer Rating", 1.0, 5.0, 4.0, 0.1)
numreviews = st.sidebar.number_input("Number of Reviews", min_value=0, max_value=50000, value=250, step=50)

st.sidebar.markdown("---")
st.sidebar.markdown("**📦 Inventory**")
stockquantity = st.sidebar.number_input("Stock Quantity", min_value=0, max_value=10000, value=150, step=10)
stock_group   = st.sidebar.selectbox("Stock Strategy", ["Limited Quantity", "Vast Quantity"])

st.sidebar.markdown("---")
st.sidebar.markdown("**🏷️ Product & Market**")
selected_category = st.sidebar.selectbox("Product Category", categories) if categories else None
selected_city     = st.sidebar.selectbox("Target City", cities)          if cities     else None

# --- MAIN LAYOUT ---
col_left, col_right = st.columns([3, 2], gap="large")

# ── LEFT: PREDICTION PANEL ────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="section-header">Forecasting Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Live Sales Prediction</div>', unsafe_allow_html=True)

    if st.button("🚀  Run Prediction Engine"):
        with st.spinner("Analysing market signals…"):
            import time; time.sleep(0.5)   # brief visual pause for UX

            # Build the input DataFrame from model features
            input_df = pd.DataFrame(0.0, index=[0], columns=model_features)

            for col in input_df.columns:
                cl = col.lower()
                if cl == 'price':         input_df[col] = price
                if cl == 'rating':        input_df[col] = rating
                if cl == 'numreviews':    input_df[col] = numreviews
                if cl == 'stockquantity': input_df[col] = stockquantity
                if cl == 'discount':      input_df[col] = discount
                if cl == 'group':
                    input_df[col] = 1 if stock_group == "Limited Quantity" else 0

            if selected_category:
                cat_col = f"category_{selected_category}"
                if cat_col in input_df.columns:
                    input_df[cat_col] = 1

            if selected_city:
                city_col = f"city_{selected_city}"
                if city_col in input_df.columns:
                    input_df[city_col] = 1

            try:
                prediction_log = model.predict(input_df)
                # Reverse log-transform applied during training (np.log1p → np.expm1)
                final_val = np.expm1(prediction_log)[0]
                final_val = max(0, final_val)  # clamp to non-negative

                # ── RESULT CARD ──
                st.success(f"### Predicted Monthly Sales\n# {int(final_val):,} units / month")
                st.markdown("---")

                # ── INSIGHT SUMMARY ──
                st.markdown("### Market Intelligence")
                st.markdown("Prediction Insights based on current market data.")

                # Revenue estimate
                revenue = final_val * price * (1 - discount / 100)
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Est. Monthly Revenue", f"${revenue:,.0f}", "After discount")
                with col_b:
                    ratio = round(final_val / max(stockquantity, 1), 1)
                    trend = "-Low Stock" if ratio > 2 else ("Moderate" if ratio > 0.8 else "Healthy")
                    st.metric("Sell-Through Ratio", f"{ratio}x", trend)
                with col_c:
                    score = min(100, int(rating * 18 + min(numreviews, 500) / 25))
                    st.metric("Social Proof Score", f"{score}/100")

            except Exception as e:
                st.error(f"**Prediction failed:** {e}")
    else:
        # Empty state
        st.info("🤖 **Ready** \n\nConfigure parameters in the sidebar, then click **Run Prediction Engine**")

# ── RIGHT: PARAMETER SUMMARY ──────────────────────────────────────────────────
with col_right:
    import textwrap
    st.markdown('<div class="section-header">Current Configuration</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Parameter Summary</div>', unsafe_allow_html=True)

    # Pre-compute all values
    urgency_color = "#ef4444" if stock_group == "Limited Quantity" else "#22c55e"
    urgency_label = "HIGH URGENCY" if stock_group == "Limited Quantity" else "STABLE SUPPLY"
    stars         = "⭐" * int(rating)
    cat_label     = selected_category or "All Categories"
    city_label    = selected_city or "All Cities"
    eff_price     = price * (1 - discount / 100)
    pct           = int((rating - 1) / 4 * 100)
    bar_color     = "#22c55e" if pct > 70 else ("#fbbf24" if pct > 40 else "#ef4444")

    st.markdown("### Market Urgency")
    if stock_group == "Limited Quantity":
        st.error("🚨 HIGH URGENCY - Low Stock!")
    else:
        st.success("✅ STABLE SUPPLY")

    st.markdown("### Parameter Summary")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Unit Price", f"${price:.2f}")
        st.metric("Effective Price", f"${eff_price:.2f}", f"-{discount:.0f}% discount")
        st.metric("Category", f"{cat_label}")
    with col2:
        st.metric("Customer Rating", f"{rating:.1f}/5.0")
        st.metric("Reviews", f"{numreviews:,}")
        st.metric("Stock Quantity", f"{stockquantity:,}", "-Limited" if stock_group == "Limited Quantity" else "")

    st.markdown("### Active Filters")
    st.info(f"📍 City: {city_label} | 📦 Category: {cat_label}")

    st.markdown("### Rating Health")
    st.progress(pct / 100.0, text=f"Customer Satisfaction: {rating:.1f}/5.0")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
    <div class="footer-card">
        <strong>&#128161; Model Note:</strong> This tool uses a <strong>Random Forest Regressor</strong> trained on
        historical e-commerce sales data. Predictions account for the <strong>Scarcity Effect</strong>
        &mdash; limited stock tends to correlate with higher conversion urgency. The model incorporates
        pricing, social proof signals, inventory levels, product category, and geographic demand patterns.
        Use predictions as strategic guidance, not guaranteed outcomes.
    </div>
""", unsafe_allow_html=True)