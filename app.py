import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

st.set_page_config(
    page_title="House Price Valuation - 76 Features",
    page_icon="🏡",
    layout="wide"
)

# ตั้งชื่อไฟล์โมเดลให้ตรงกับที่คุณอัปโหลด
MODEL_PATH = "linear_regression_model.pkl" 

# รายชื่อ 76 Features ตามลำดับที่โมเดลต้องการเป๊ะๆ
FEATURE_NAMES = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', 
    '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 
    'GarageCars', 'GarageArea', 'MoSold', 'HalfBath_binned', 'BsmtFullBath_binned', 
    'Fireplaces_binned', 'TotalPorchSF', 'MasVnrArea_binary', 'OpenPorchSF_binary', 
    'WoodDeckSF_binary', 'HouseAge', 'RemodAge', 'IsRemodeled',
    'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 
    'LotShape_IR2', 'LotShape_IR3', 'LotShape_Reg',
    'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_FR3', 'LotConfig_Inside',
    'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 
    'HouseStyle_2Story', 'HouseStyle_SFoyer', 'HouseStyle_SLvl',
    'RoofStyle_Gable', 'RoofStyle_Gambrel', 'RoofStyle_Hip', 'RoofStyle_Mansard', 'RoofStyle_Shed',
    'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone',
    'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA',
    'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Foundation_Stone', 'Foundation_Wood',
    'GarageType_Attchd', 'GarageType_Basment', 'GarageType_BuiltIn', 'GarageType_CarPort', 'GarageType_Detchd', 'GarageType_NoGarage',
    'Neighborhood', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageFinish'
]

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"ไม่พบไฟล์ '{MODEL_PATH}'"
    try:
        loaded_model = joblib.load(MODEL_PATH)
        return loaded_model, "โหลดโมเดลสำเร็จ (76 Features)"
    except Exception as e:
        return None, f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}"

model, model_status = load_model()

st.title("🏡 Advanced House Price Prediction (Ames Housing)")
if model is not None:
    st.success(f"✅ สถานะโมเดล: `{model_status}`")
else:
    st.error(f"❌ สถานะโมเดล: `{model_status}`")

col_input, col_result = st.columns([1.2, 0.8], gap="large")

with col_input:
    st.subheader("📋 ระบุคุณลักษณะของบ้าน")
    
    with st.expander("1. พื้นที่และขนาด (Size & Area)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            gr_liv_area = st.number_input("พื้นที่ใช้สอยรวม (ตารางฟุต - GrLivArea)", min_value=300, max_value=10000, value=1500, step=50)
            lot_area = st.number_input("ขนาดที่ดิน (ตารางฟุต - LotArea)", min_value=1000, max_value=50000, value=10000, step=100)
        with c2:
            first_flr_sf = st.number_input("พื้นที่ชั้น 1 (1stFlrSF)", min_value=300, max_value=5000, value=1000, step=50)
            second_flr_sf = st.number_input("พื้นที่ชั้น 2 (2ndFlrSF)", min_value=0, max_value=5000, value=500, step=50)
        with c3:
            total_bsmt_sf = st.number_input("พื้นที่ห้องใต้ดินรวม (TotalBsmtSF)", min_value=0, max_value=5000, value=1000, step=50)
            garage_area = st.number_input("พื้นที่โรงจอดรถ (GarageArea)", min_value=0, max_value=2000, value=400, step=50)

    with st.expander("2. จำนวนห้องและฟังก์ชัน (Rooms & Amenities)", expanded=True):
        c4, c5, c6 = st.columns(3)
        with c4:
            bedroom = st.slider("ห้องนอน (BedroomAbvGr)", 0, 8, 3)
            tot_rms = st.slider("ห้องทั้งหมดไม่รวมห้องน้ำ (TotRms)", 2, 14, 6)
        with c5:
            full_bath = st.slider("ห้องน้ำเต็มรูปแบบ (FullBath)", 0, 4, 2)
            half_bath = st.checkbox("มีห้องน้ำเล็ก (HalfBath)")
        with c6:
            garage_cars = st.slider("ความจุจอดรถ (คัน)", 0, 5, 2)
            has_fireplace = st.checkbox("มีเตาผิง (Fireplace)")
            has_wood_deck = st.checkbox("มีระเบียงไม้ (WoodDeck)")

    with st.expander("3. อายุบ้านและทำเล (Age & Location)", expanded=True):
        c7, c8 = st.columns(2)
        with c7:
            house_age = st.number_input("อายุของบ้าน (ปี - HouseAge)", min_value=0, max_value=150, value=15)
            is_remodeled = st.checkbox("เคยได้รับการรีโนเวท (IsRemodeled)", value=True)
            remod_age = st.number_input("อายุหลังจากการรีโนเวท (ปี)", min_value=0, max_value=150, value=10, disabled=not is_remodeled)
        with c8:
            neighborhood = st.slider("รหัสย่านที่ตั้ง (Neighborhood: 0-25)", 0, 25, 12)
            ms_zoning = st.selectbox("โซนผังเมือง (MSZoning)", ["RL", "RM", "FV", "RH"])

    with st.expander("4. เกรดและคุณภาพวัสดุ (Quality Ratings)", expanded=True):
        qual_options = {"Excellent": 5.0, "Good": 4.0, "Typical": 3.0, "Fair": 2.0, "Poor": 1.0}
        c9, c10 = st.columns(2)
        with c9:
            kitchen_qual = st.selectbox("คุณภาพห้องครัว", list(qual_options.keys()), index=2)
            exter_qual = st.selectbox("คุณภาพวัสดุภายนอก", list(qual_options.keys()), index=2)
        with c10:
            heating_qc = st.selectbox("คุณภาพระบบทำความร้อน", list(qual_options.keys()), index=0) # มักจะ Ex (5)
            bsmt_qual = st.selectbox("คุณภาพห้องใต้ดิน", list(qual_options.keys()), index=2)

    btn_predict = st.button("🔮 คำนวณราคาประเมิน (Predict)", type="primary", use_container_width=True)

with col_result:
    st.subheader("📊 ผลการประเมินราคา")
    
    if btn_predict:
        if model is None:
            st.error("โมเดลไม่พร้อมใช้งาน กรุณาตรวจสอบไฟล์ .pkl")
        else:
            # 1. เตรียมค่าเริ่มต้นเป็น 0 สำหรับทุก 76 คอลัมน์
            row_data = {col: 0.0 for col in FEATURE_NAMES}

            # 2. ใส่ค่าตัวแปรเชิงปริมาณ (Numeric)
            row_data['GrLivArea'] = float(gr_liv_area)
            row_data['LotArea'] = float(lot_area)
            row_data['1stFlrSF'] = float(first_flr_sf)
            row_data['2ndFlrSF'] = float(second_flr_sf)
            row_data['TotalBsmtSF'] = float(total_bsmt_sf)
            row_data['GarageArea'] = float(garage_area)
            row_data['BedroomAbvGr'] = float(bedroom)
            row_data['TotRmsAbvGrd'] = float(tot_rms)
            row_data['FullBath'] = float(full_bath)
            row_data['GarageCars'] = float(garage_cars)
            row_data['HouseAge'] = float(house_age)
            row_data['IsRemodeled'] = 1.0 if is_remodeled else 0.0
            row_data['RemodAge'] = float(remod_age) if is_remodeled else float(house_age)
            
            # Binary & Binned Features
            row_data['HalfBath_binned'] = 1.0 if half_bath else 0.0
            row_data['Fireplaces_binned'] = 1.0 if has_fireplace else 0.0
            row_data['WoodDeckSF_binary'] = 1.0 if has_wood_deck else 0.0
            
            # Default numeric ให้บ้านมีค่ากลางๆ เพื่อไม่ให้โมเดลประเมินเพี้ยน
            row_data['MSSubClass'] = 20.0  # 1-Story
            row_data['LotFrontage'] = 70.0
            row_data['MoSold'] = 6.0       # เดือนมิถุนายน
            
            # 3. One-Hot Encoding พื้นฐาน
            if ms_zoning == "RL": row_data['MSZoning_RL'] = 1.0
            elif ms_zoning == "RM": row_data['MSZoning_RM'] = 1.0
            elif ms_zoning == "FV": row_data['MSZoning_FV'] = 1.0
            elif ms_zoning == "RH": row_data['MSZoning_RH'] = 1.0
            
            # Default One-Hot ที่คนใช้บ่อยที่สุดของ Ames
            row_data['LotShape_Reg'] = 1.0
            row_data['LotConfig_Inside'] = 1.0
            row_data['HouseStyle_1Story'] = 1.0 if second_flr_sf == 0 else 0.0
            row_data['HouseStyle_2Story'] = 1.0 if second_flr_sf > 0 else 0.0
            row_data['RoofStyle_Gable'] = 1.0
            row_data['Foundation_PConc'] = 1.0
            row_data['GarageType_Attchd'] = 1.0
            row_data['MasVnrType_None'] = 1.0
            
            # 4. Ordinal Features & Quality
            row_data['Neighborhood'] = float(neighborhood)
            row_data['Exterior1st'] = 12.0
            row_data['Exterior2nd'] = 13.0
            row_data['KitchenQual'] = qual_options[kitchen_qual]
            row_data['ExterQual'] = qual_options[exter_qual]
            row_data['HeatingQC'] = qual_options[heating_qc]
            row_data['BsmtQual'] = qual_options[bsmt_qual] if total_bsmt_sf > 0 else 0.0
            row_data['FireplaceQu'] = 3.0 if has_fireplace else 0.0
            row_data['GarageFinish'] = 2.0 if garage_cars > 0 else 0.0
            row_data['BsmtExposure'] = 1.0
            row_data['BsmtFinType1'] = 4.0

            # 5. สร้าง DataFrame ให้มีลำดับเป๊ะๆ ตาม FEATURE_NAMES
            input_df = pd.DataFrame([row_data])[FEATURE_NAMES]

            try:
                pred = model.predict(input_df)
                price = float(pred[0])

                if price >= 0:
                    st.metric(label="ราคาประเมิน (Estimated Price)", value=f"${price:,.2f}")
                else:
                    st.metric(label="ราคาประเมิน (Estimated Price)", value=f"-${abs(price):,.2f}")
                    st.warning("⚠️ ราคาคำนวณติดลบ: อาจเป็นเพราะตั้งค่าอายุบ้านสูงไป หรือพื้นที่ใช้สอยน้อยเกินไป")

                with st.expander("ดูตาราง 76 Features (ที่ส่งเข้าโมเดล)"):
                    st.dataframe(input_df.T, height=400)

            except Exception as e:
                st.error(f"Prediction Error: {e}")
    else:
        st.info("👈 ระบุรายละเอียดพื้นที่บ้านทางด้านซ้าย แล้วกดปุ่มเพื่อเริ่มประเมินราคา")
