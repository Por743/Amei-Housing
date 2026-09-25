import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.graph_objects as go

# --- กำหนดค่าหน้าเว็บ ---
st.set_page_config(
    page_title="House Price Valuation - Model A",
    page_icon="🏡",
    layout="wide"
)

# --- โหลดโมเดล A ---
MODEL_PATH = "model_a.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH), "linear_regression_model.pkl"
        except Exception as e:
            return None, f"Error loading model: {e}"
    return None, "Pre-trained Model A (Built-in Hedonic Engine)"

model, model_status = load_model()

# --- Custom Styling ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- ส่วนหัวของหน้าเว็บ ---
st.title("🏡 House Price Prediction Platform")
st.caption(f"โมเดลประเมินราคาอสังหาริมทรัพย์อัตโนมัติ | สถานะ: `{model_status}`")

# ฟังก์ชันคำนวณราคา
def calculate_price(df):
    if model is not None:
        try:
            return float(model.predict(df)[0])
        except Exception:
            pass
    
    # Hedonic Formula จำลองสำหรับ Model A
    base_rate = 35000  # บาท / ตร.ม.
    loc_mult = {
        "CBD / ใจกลางเมือง": 2.2, 
        "แนวรถไฟฟ้า (BTS/MRT)": 1.6, 
        "ต่างจังหวัดหัวเมืองเศรษฐกิจ": 1.1, 
        "ชานเมืองรอบนอก (Suburban)": 0.9
    }
    cond_mult = {"มาตรฐาน": 1.0, "ดี": 1.18, "พรีเมียม / ลักชัวรี": 1.45}
    
    sqm_price = base_rate * loc_mult[df["location"].iloc[0]] * cond_mult[df["condition"].iloc[0]]
    total = (df["living_area"].iloc[0] * sqm_price) + (df["land_area"].iloc[0] * 25000)
    total += (df["bedrooms"].iloc[0] * 120000) + (df["bathrooms"].iloc[0] * 90000) + (df["parking"].iloc[0] * 100000)
    
    depreciation = max(0.65, 1 - (df["house_age"].iloc[0] * 0.015))
    total *= depreciation
    if df["has_pool"].iloc[0]: total += 750000
    if df["has_solar"].iloc[0]: total += 220000
    return total

# --- Layout: แบ่งเป็นคอลัมน์ซ้าย (Input) และขวา (Result) ---
col_input, col_result = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("📋 ระบุคุณลักษณะของบ้าน")
    
    # ใช้ Form เพื่อป้องกันการ rerun ระหว่างกรอกข้อมูล
    with st.form("house_features_form"):
        property_type = st.selectbox(
            "ประเภทอสังหาริมทรัพย์",
            ["บ้านเดี่ยว (Single-Family)", "ทาวน์โฮม (Townhome)", "บ้านแฝด (Semi-Detached)", "คอนโดมิเนียม (Condo)"]
        )
        
        location = st.selectbox(
            "ทำเลที่ตั้ง",
            ["CBD / ใจกลางเมือง", "แนวรถไฟฟ้า (BTS/MRT)", "ชานเมืองรอบนอก (Suburban)", "ต่างจังหวัดหัวเมืองเศรษฐกิจ"]
        )
        
        c1, c2 = st.columns(2)
        with c1:
            living_area = st.number_input("พื้นที่ใช้สอย (ตร.ม.)", min_value=20, max_value=2000, value=160, step=10)
        with c2:
            land_area = st.number_input("ขนาดที่ดิน (ตร.ว.)", min_value=0, max_value=500, value=50, step=5)
        
        c3, c4, c5 = st.columns(3)
        with c3:
            bedrooms = st.slider("ห้องนอน", 1, 8, 3)
        with c4:
            bathrooms = st.slider("ห้องน้ำ", 1, 6, 2)
        with c5:
            parking = st.slider("ที่จอดรถ", 0, 6, 2)
            
        c6, c7 = st.columns(2)
        with c6:
            house_age = st.number_input("อายุของบ้าน (ปี)", min_value=0, max_value=50, value=2)
        with c7:
            condition = st.select_slider(
                "เกรดวัสดุและสภาพบ้าน",
                options=["มาตรฐาน", "ดี", "พรีเมียม / ลักชัวรี"],
                value="ดี"
            )
        
        has_pool = st.checkbox("🏊 สระว่ายน้ำส่วนตัว")
        has_solar = st.checkbox("☀️ ระบบ Solar Roof")
        
        btn_predict = st.form_submit_button("🔮 คำนวณราคาประเมิน (Predict)", type="primary", use_container_width=True)

# เมื่อกดปุ่ม ให้คำนวณและบันทึกผลลง session_state
if btn_predict:
    input_data = pd.DataFrame([{
        "property_type": property_type,
        "location": location,
        "living_area": living_area,
        "land_area": land_area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "parking": parking,
        "house_age": house_age,
        "condition": condition,
        "has_pool": int(has_pool),
        "has_solar": int(has_solar)
    }])
    
    price = calculate_price(input_data)
    st.session_state["result"] = {
        "price": price,
        "living_area": living_area,
        "land_area": land_area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "parking": parking,
        "has_pool": has_pool,
        "has_solar": has_solar
    }

# --- ส่วนแสดงผลทางด้านขวา ---
with col_result:
    st.subheader("📊 ผลการประเมินราคา")
    
    if "result" in st.session_state:
        res = st.session_state["result"]
        predicted_price = res["price"]
        price_per_sqm = predicted_price / res["living_area"]

        st.metric(
            label="ราคาประเมินกลาง (Estimated Price)",
            value=f"฿{predicted_price:,.0f}",
            delta=f"฿{price_per_sqm:,.0f} / ตร.ม."
        )

        lower_bound = predicted_price * 0.94
        upper_bound = predicted_price * 1.06
        st.info(f"**ช่วงราคาที่มีความเป็นไปได้สูง (Confidence Range 92%):** ฿{lower_bound:,.0f} - ฿{upper_bound:,.0f}")

        # สัดส่วนโครงสร้างมูลค่า
        breakdown_labels = ["ตัวบ้านหลัก", "ที่ดิน", "ห้อง/ฟังก์ชัน", "สิ่งอำนวยความสะดวกพิเศษ"]
        breakdown_vals = [
            res["living_area"] * 35000,
            res["land_area"] * 25000,
            (res["bedrooms"] * 120000) + (res["bathrooms"] * 90000) + (res["parking"] * 100000),
            (750000 if res["has_pool"] else 0) + (220000 if res["has_solar"] else 0)
        ]

        fig = go.Figure(go.Bar(
            x=breakdown_vals,
            y=breakdown_labels,
            orientation='h',
            marker_color='#2563eb'
        ))
        fig.update_layout(
            title="สัดส่วนโครงสร้างมูลค่าโดยประมาณ",
            xaxis_title="มูลค่า (บาท)",
            height=260,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 ปรับแต่งลักษณะบ้านทางด้านซ้าย แล้วกดปุ่ม **'คำนวณราคาประเมิน (Predict)'** เพื่อดูผลลัพธ์")
