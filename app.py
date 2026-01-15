# =========================
# 0. 页面配置（必须最先）
# =========================
import streamlit as st
st.set_page_config(
    page_title="High Compensation Ratio Risk Prediction Model",
    layout="wide"
)

# =========================
# 1. 依赖
# =========================
import pandas as pd
import shap
import joblib
import tempfile
import os
import streamlit.components.v1 as components

# =========================
# 2. 加载模型
# =========================
model = joblib.load("model.pkl")
st.write(model)
features = list(model.feature_names_in_)

explainer = shap.TreeExplainer(model)

# =========================
# 3. 页面标题
# =========================
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>"
    "High Compensation Ratio Risk Prediction Model"
    "</h1>",
    unsafe_allow_html=True
)

# =========================
# 4. 输入区域
# =========================
col1, col2 = st.columns(2)

with col1:
    Gender = st.selectbox("Gender", ["Male (1)", "Female (0)"])
    Age = st.number_input("Age", 0, 120, 40)
    Disability_severity_grade = st.selectbox(
        "Disability severity grade", ["Yes (1)", "No (0)"]
    )
    Inappropriate_surgical_procedure = st.selectbox(
        "Inappropriate surgical procedure", ["Yes (1)", "No (0)"]
    )

with col2:
    Inadequate_medical_records = st.selectbox(
        "Inadequate medical records", ["Yes (1)", "No (0)"]
    )
    Lack_of_informed_consent = st.selectbox(
        "Lack of informed consent", ["Yes (1)", "No (0)"]
    )
    Treatment_delay = st.selectbox(
        "Treatment delay", ["Yes (1)", "No (0)"]
    )
    Inadequate_preoperative_preparation = st.selectbox(
        "Inadequate preoperative preparation", ["Yes (1)", "No (0)"]
    )

# =========================
# 5. 数值化（预测样本）
# =========================
X_input = pd.DataFrame([{
    "Gender": 1 if "Male" in Gender else 0,
    "Age": Age,
    "Disability_severity_grade": 1 if "Yes" in Disability_severity_grade else 0,
    "Inappropriate_surgical_procedure": 1 if "Yes" in Inappropriate_surgical_procedure else 0,
    "Inadequate_medical_records": 1 if "Yes" in Inadequate_medical_records else 0,
    "Lack_of_informed_consent": 1 if "Yes" in Lack_of_informed_consent else 0,
    "Treatment_delay": 1 if "Yes" in Treatment_delay else 0,
    "Inadequate_preoperative_preparation": 1 if "Yes" in Inadequate_preoperative_preparation else 0
}])[features]

# =========================
# 6. 预测按钮
# =========================
st.markdown("---")
predict_btn = st.button("🔮 Predict Risk", use_container_width=True)

# =========================
# 7. 预测 + 结果展示 + SHAP
# =========================
if predict_btn:

    # -------- 预测结果 --------
    st.write(X_input)
    prob = model.predict_proba(X_input)[0][1]
    pred = model.predict(X_input)[0]

    st.subheader("📊 Prediction Result")

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.metric(
            label="Predicted Risk Probability",
            value=f"{prob:.3f}"
        )

    with col_r2:
        st.metric(
            label="Predicted Risk Level",
            value="High Risk" if pred == 1 else "Low Risk"
        )

    # -------- SHAP --------
    st.markdown("---")
    st.subheader("🔍 SHAP Explanation (Force Plot)")

    shap_values = explainer.shap_values(X_input)
    base_value = explainer.expected_value

    shap_html = shap.plots.force(
        base_value[1],
        shap_values[1][0],
        feature_names=X_input.columns,
        matplotlib=False
    )

    html_content = f"""
    <head>{shap.getjs()}</head>
    <body>
        <div style="text-align:center; font-size:16px; font-weight:600;">
            Contribution of each feature to the predicted high compensation risk
        </div>
        {shap_html.html()}
    </body>
    """

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".html",
        mode="w",
        encoding="utf-8"
    ) as tmp_file:
        tmp_file.write(html_content)
        tmp_path = tmp_file.name

    with open(tmp_path, "r", encoding="utf-8") as f:
        components.html(f.read(), height=300)

    os.remove(tmp_path)





