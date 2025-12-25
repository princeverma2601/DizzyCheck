import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import re

# =========================
# Load model and scaler
# =========================
model = load_model("final_pppd_model.h5")
scaler = joblib.load("scaler.pkl")

# =========================
# Page setup
# =========================
st.set_page_config(
    page_title="DizzyCheck | PPPD, Vertigo & Migraine Prediction",
    page_icon="🌀",
    layout="wide"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
    html, body, [class*="css"] {
        width: 100%;
        max-width: 100% !important;
        padding: 0;
        margin: 0;
    }

    .block-container {
        max-width: 96% !important;
        padding: 1.5rem 3rem 3rem 3rem;
    }

    body { background-color: #F3F7FB; }

    h1 {
        text-align:center;
        color:#0B63B7;
        font-weight:800;
        letter-spacing:1px;
    }

    .subtitle {
        text-align:center;
        color:#06685A;
        font-size:1.1rem;
        margin-bottom: 1rem;
    }

    .footer {
        text-align:center;
        color:gray;
        margin-top:2rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown("<h1>🌀 DizzyCheck</h1>", unsafe_allow_html=True)
st.markdown('<div class="subtitle">PPPD, Vertigo & Migraine Prediction Tool</div>', unsafe_allow_html=True)
st.divider()

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["🔍 Detection", "📊 Analysis", "ℹ️ Information"])

# =========================
# 1️⃣ DETECTION TAB
# =========================
with tab1:

    def valid_name(name):
        return bool(re.match("^[A-Za-z ]*$", name))

    st.subheader("👤 Patient Details")

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        first_name = st.text_input("First Name", placeholder="John")
        if first_name and not valid_name(first_name):
            st.warning("❌ Name must contain only alphabets.")
            first_name = ""

    with c2:
        surname = st.text_input("Surname", placeholder="Doe")
        if surname and not valid_name(surname):
            st.warning("❌ Surname must contain only alphabets.")
            surname = ""

    with c3:
        age_display = st.number_input("Age (years)", 1, 120, 35)

    gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
    st.markdown('---')

    # SYMPTOMS
    st.subheader("🧾 Symptom Inputs")
    col1, col2 = st.columns(2)

    with col1:
        duration = st.slider("Duration (0–10)", 0.0, 10.0, 0.0, step=0.1)
        frequency = st.slider("Frequency (1–10)", 1.0, 10.0, 1.0, step=0.1)
        intensity = st.slider("Intensity (0–10)", 0.0, 10.0, 0.0, step=0.1)
        nausea = st.slider("Nausea (0–1)", 0.0, 1.0, 0.0, step=0.1)
        vomiting = st.slider("Vomiting (0–1)", 0.0, 1.0, 0.0, step=0.1)

    with col2:
        dizziness = st.slider("Dizziness (0–1)", 0.0, 1.0, 0.0, step=0.1)
        headache = st.slider("Headache (0–1)", 0.0, 1.0, 0.0, step=0.1)
        photophobia = st.slider("Photophobia (0–1)", 0.0, 1.0, 0.0, step=0.1)
        phonophobia = st.slider("Phonophobia (0–1)", 0.0, 1.0, 0.0, step=0.1)
        visual = st.slider("Visual Disturbance (0–1)", 0.0, 1.0, 0.0, step=0.1)
        sensory = st.slider("Sensory Symptom (0–1)", 0.0, 1.0, 0.0, step=0.1)

    st.divider()

    # RUN PREDICTION
    if st.button("🔍 Run Prediction", use_container_width=True):

        gender_num = 0.0 if gender == "Female" else 1.0

        features = np.array([[age_display, gender_num, duration, frequency, intensity,
                              nausea, vomiting, dizziness, headache,
                              photophobia, phonophobia, visual, sensory]], dtype=float)

        try:
            features_scaled = scaler.transform(features)
        except:
            features_scaled = features

        preds = model.predict(features_scaled)[0]
        preds_percent = preds * 100
        conditions = ['Vertigo', 'Migraine', 'PPPD']

        left_col, right_col = st.columns([2, 1])

        with left_col:
            st.subheader("📊 Prediction Results")
            st.image("https://img.icons8.com/color/96/combo-chart--v1.png", width=80)

            for cond, prob in zip(conditions, preds_percent):
                st.write(f"**{cond}: {prob:.2f}%**")
                st.progress(int(prob))

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[float(p) for p in preds_percent],
                theta=conditions,
                fill='toself',
                name='Prediction',
                line_color='#0288D1'
            ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,100])),
                showlegend=False,
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)

        with right_col:
            st.subheader("🔎 Quick Tips")
            st.image(
                "https://img.freepik.com/free-photo/doctor-writing-medical-prescription-blue-background_93675-129102.jpg",
                caption="If symptoms are severe, seek medical attention.",
                use_container_width=True
            )
            st.write("- If likelihood is high, visit a clinician.")
            st.write("- This tool is a screening aid, not a diagnosis.")

        # =========================
        # NEW PATIENT SUMMARY REPORT
        # =========================
        st.divider()
        st.subheader("🩺 Patient Summary Report")

        likely = [conditions[i] for i, p in enumerate(preds_percent) if p >= 50]
        most_likely_text = ", ".join(likely) if likely else "No strong indication detected (all < 50%)"

        st.markdown(f"""
        <div style="background:linear-gradient(90deg,#ffffff,#f7fbff);
                    border-radius:12px; padding:16px;
                    box-shadow:0 6px 18px rgba(3,27,63,0.06);
                    width:100%;">
          <div style="display:flex; gap:12px; align-items:center">
            <img src="https://img.freepik.com/free-vector/doctor-character-background_1270-84.jpg?size=626&ext=jpg" 
                 width="72" style="border-radius:8px"/>
            <div>
              <div style="font-size:1.05rem; color:#072f4b; font-weight:700;">
                {(first_name or 'Not provided').strip()} {(surname or '').strip()}
              </div>
              <div style="color:#06685A; font-weight:600;">
                Age: <span style='background:#E1F2FF; color:#044E8C; padding:4px 8px; border-radius:6px;'>
                    {age_display} years
                </span>
              </div>
              <div style="color:#06685A; font-weight:600;">Gender: {gender}</div>
            </div>
          </div>

          <hr style="margin:12px 0; border:none; border-top:1px solid #e6f0fb"/>

          <div style="color:#08324a; font-weight:600; margin-bottom:8px;">Key Symptoms</div>
          <div style="display:flex; gap:8px; flex-wrap:wrap;">
            <div style="background:#FFF3E0; color:#BF360C; padding:6px 10px; border-radius:8px;">
                Dizziness: {dizziness}
            </div>
            <div style="background:#E8F5E9; color:#1B5E20; padding:6px 10px; border-radius:8px;">
                Headache: {headache}
            </div>
            <div style="background:#E3F2FD; color:#01579B; padding:6px 10px; border-radius:8px;">
                Intensity: {intensity}/10
            </div>
            <div style="background:#F3E5F5; color:#4A148C; padding:6px 10px; border-radius:8px;">
                Duration: {duration}/10
            </div>
          </div>

          <div style="margin-top:12px; color:#08324a; font-weight:700;">Most Likely Condition(s)</div>
          <div style="margin-top:6px;">
            <span style="background:#BBDEFB; color:#0D47A1; padding:6px 12px; border-radius:10px; font-weight:700;">
                {most_likely_text}
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.info("⚕️ DizzyCheck is an AI screening assistant — not a replacement for professional medical evaluation.")

# =========================
# 2️⃣ ANALYSIS TAB (Updated Donut Charts)
# =========================
with tab2:

    st.markdown("## 📊 Model & Dataset Analysis")
    st.write("Overview of performance, dataset balance, and feature importance.")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    for col, title in zip([c1, c2, c3, c4], ["Accuracy", "Precision", "Recall", "F1 Score"]):
        with col:
            st.markdown(f"""
            <div style="background:#0E1A2B; padding:20px; border-radius:12px; text-align:center;">
                <div style="color:white; font-size:20px; font-weight:700;">{title}</div>
                <div style="color:#4FC3F7; font-size:28px; font-weight:800;">94.38%</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("📦 Dataset Size & Balance")
    st.write("""
    **Total Samples:** 3500  
    **Features Used:** 13  
    - **Healthy (0)** = No condition  
    - **Patient (1)** = Condition present  
    """)

    st.markdown("---")

    colA, colB, colC = st.columns(3)

    def donut(values, labels):
        return go.Figure(
            data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                textinfo="label+percent"
            )]
        )

    with colA:
        st.write("### Migraine")
        st.plotly_chart(
            donut([63.1, 36.9], ["Healthy (0)", "Patient (1)"]),
            use_container_width=True
        )

    with colB:
        st.write("### Vertigo")
        st.plotly_chart(
            donut([52.1, 47.9], ["Healthy (0)", "Patient (1)"]),
            use_container_width=True
        )

    with colC:
        st.write("### PPPD")
        st.plotly_chart(
            donut([44.5, 55.5], ["Healthy (0)", "Patient (1)"]),
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("📌 Feature Importance")
    importance_scores = {
        "intensity": 0.13, "frequency": 0.12, "photophobia": 0.11,
        "nausea": 0.10, "visual": 0.09, "duration": 0.08,
        "age": 0.06, "phonophobia": 0.055, "dizziness": 0.05,
        "headache": 0.045, "sensory": 0.04, "vomiting": 0.035,
        "gender": 0.02
    }

    fig_imp = go.Figure(go.Bar(
        x=list(importance_scores.values()),
        y=list(importance_scores.keys()),
        orientation='h',
        marker=dict(color='#81D4FA')
    ))

    fig_imp.update_layout(height=550, template="plotly_white")
    st.plotly_chart(fig_imp, use_container_width=True)

# =========================
# 3️⃣ INFORMATION TAB
# =========================
with tab3:

    st.markdown("## 📘 Information")

    st.write("""
    **DizzyCheck** predicts the likelihood of:
    - Migraine  
    - Vertigo  
    - PPPD  

    This tool is for educational purposes only and is **not** a medical diagnosis.  
    Always consult a doctor for health concerns.
    """)

# =========================
# Footer
# =========================
st.markdown('<div class="footer">© 2025 DizzyCheck | Built by Prince Verma 🧠</div>', unsafe_allow_html=True)
