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

# --- โหลดโมเดลจริง ---
MODEL_PATH = "linear_regression_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"ไม่พบไฟล์ '{MODEL_PATH}' ในโฟลเดอร์โปรเจกต์"
    try:
        loaded = joblib.load(MODEL_PATH)
        return loaded, f"โหลดสำเร็จ: {MODEL_PATH}"
    except Exception as e:
        return None, f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}"

model, model_status = load_model()

# ดึงชื่อคอลัมน์ที่โมเดลเคยใช้ตอนเทรน (ถ้ามี)
expected_features = []
if model is not None and hasattr(model, "feature_names_in_"):
    expected_features = list(model.feature_names_in_)

# --- ส่วนหัวของหน้าเว็บ ---
st.title("🏡 House Price Prediction Platform")
if model is not None:
    st.success(f"สถานะโมเดล: `{model_status}`")
    if expected_features:
        st.caption(f"โมเดลต้องการ Features: `{', '.join(expected_features)}`")
else:
    st.error(f"สถานะโมเดล: `{model_status}`")

# --- Layout: ซ้าย (Input) และ ขวา (Result) ---
col_input, col_result = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("📋 ระบุคุณลักษณะของบ้าน")
    
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

# ฟังก์ชันจัดเตรียมข้อมูลส่งเข้า Model จริง
def prepare_input_for_model(raw_dict, model_obj):
    df_raw = pd.DataFrame([raw_dict])
    
    # กรณีโมเดลบันทึก feature_names_in_ ไว้ตอนเทรน
    if hasattr(model_obj, "feature_names_in_"):
        req_cols = list(model_obj.feature_names_in_)
        
        # แมปปิ้งชื่อคอลัมน์ทั่วไปที่มักใช้ในการเทรน
        alias_map = {
            "area": living_area,
            "sqm": living_area,
            "size": living_area,
            "living_area": living_area,
            "land_area": land_area,
            "land": land_area,
            "beds": bedrooms,
            "bedroom": bedrooms,
            "bedrooms": bedrooms,
            "baths": bathrooms,
            "bathroom": bathrooms,
            "bathrooms": bathrooms,
            "parking": parking,
            "garage": parking,
            "age": house_age,
            "house_age": house_age,
            "pool": int(has_pool),
            "solar": int(has_solar)
        }
        
        data_to_feed = {}
        for col in req_cols:
            if col in raw_dict:
                data_to_feed[col] = raw_dict[col]
            elif col.lower() in alias_map:
                data_to_feed[col] = alias_map[col.lower()]
            else:
                # ถ้าโมเดลต้องการคอลัมน์ที่ไม่รู้จัก ให้ใส่ 0 ไว้ก่อนเพื่อกัน Error
                data_to_feed[col] = 0
                
        return pd.DataFrame([data_to_feed])[req_cols]
    
    # หากโมเดลไม่มี feature_names_in_ ให้ส่งเฉพาะตัวแปรตัวเลขหลัก
    return pd.DataFrame([{
        "living_area": living_area,
        "land_area": land_area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "parking": parking,
        "house_age": house_age
    }])

# เมื่อกดปุ่ม คำนวณด้วยโมเดลจริง
if btn_predict:
    if model is None:
        st.session_state["error"] = "ไม่สามารถทำนายได้ เนื่องจากยังไม่ได้วางไฟล์ 'linear_regression_model.pkl'"
    else:
        raw_features = {
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
        }
        
        try:
            X_feed = prepare_input_for_model(raw_features, model)
            # ทำนายผลจาก Model จริง
            pred = model.predict(X_feed)
            pred_value = float(pred[0])
            
            st.session_state["error"] = None
            st.session_state["result"] = {
                "price": pred_value,
                "living_area": living_area,
                "X_feed": X_feed
            }
        except Exception as e:
            st.session_state["result"] = None
            st.session_state["error"] = f"เกิดข้อผิดพลาดจากตัวโมเดล: {e}"

# --- แสดงผลลัพธ์ทางด้านขวา ---
with col_result:
    st.subheader("📊 ผลการประเมินราคา (จาก Model จริง)")
    
    if st.session_state.get("error"):
        st.error(st.session_state["error"])
        st.info("💡 **คำแนะนำ:** ตรวจสอบว่าตอนสร้างโมเดล ใช้คอลัมน์ตัวแปรชื่ออะไรบ้าง และต้องเป็นตัวเลขทั้งหมดหรือไม่")
        
    elif "result" in st.session_state and st.session_state["result"] is not None:
        res = st.session_state["result"]
        predicted_price = res["price"]
        price_per_sqm = predicted_price / res["living_area"] if res["living_area"] > 0 else 0

        # แสดงราคาจาก Model จริง
        st.metric(
            label="ราคาประเมินจากโมเดล (Predicted Price)",
            value=f"฿{predicted_price:,.2f}",
            delta=f"฿{price_per_sqm:,.2f} / ตร.ม."
        )

        # สัมประสิทธิ์ (Coefficients) ของ Linear Regression ถ้ามี
        if hasattr(model, "coef_"):
            with st.expander("📈 ดูค่าน้ำหนักสัมประสิทธิ์ของโมเดล (Model Coefficients)"):
                coef_df = pd.DataFrame({
                    "Feature": res["X_feed"].columns,
                    "Coefficient (น้ำหนัก)": model.coef_ if len(model.coef_.shape) == 1 else model.coef_[0]
                })
                st.dataframe(coef_df, use_container_width=True)
                
                # กราฟแท่งแสดงอิทธิพลของแต่ละตัวแปรจากโมเดลจริง
                fig = go.Figure(go.Bar(
                    x=coef_df["Coefficient (น้ำหนัก)"],
                    y=coef_df["Feature"],
                    orientation='h',
                    marker_color='#2563eb'
                ))
                fig.update_layout(
                    title="ค่าสัมประสิทธิ์ (Weights) ของแต่ละ Feature",
                    xaxis_title="Coefficient Value",
                    height=260,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("🔍 ดูตาราง Data ที่ส่งเข้า `model.predict()`"):
            st.dataframe(res["X_feed"])
            
    else:
        st.info("👈 ปรับแต่งลักษณะบ้านทางด้านซ้าย แล้วกดปุ่ม **'คำนวณราคาประเมิน (Predict)'** เพื่อให้โมเดลประมวลผล")
