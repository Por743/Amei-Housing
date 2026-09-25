import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

st.set_page_config(
    page_title="House Price Valuation (Linear Regression)",
    page_icon="🏡",
    layout="wide"
)

MODEL_PATH = "linear_regression_model.pkl"

# รายชื่อ 51 Features ตามลำดับที่โมเดลต้องการ
FEATURE_NAMES = [
    'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM',
    'LotShape_IR2', 'LotShape_IR3', 'LotShape_Reg',
    'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_FR3', 'LotConfig_Inside',
    'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'HouseStyle_SLvl',
    'RoofStyle_Gable', 'RoofStyle_Gambrel', 'RoofStyle_Hip', 'RoofStyle_Mansard', 'RoofStyle_Shed',
    'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone',
    'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA',
    'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Foundation_Stone', 'Foundation_Wood',
    'GarageType_Attchd', 'GarageType_Basment', 'GarageType_BuiltIn', 'GarageType_CarPort', 'GarageType_Detchd', 'GarageType_NoGarage',
    'Neighborhood', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'BsmtQual',
    'BsmtExposure', 'BsmtFinType1', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageFinish'
]

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"ไม่พบไฟล์ '{MODEL_PATH}' ในโฟลเดอร์เดียวกันกับ app.py"
    try:
        loaded_model = joblib.load(MODEL_PATH)
        return loaded_model, "โหลดโมเดลสำเร็จ"
    except Exception as e:
        return None, f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}"

model, model_status = load_model()

st.title("🏡 House Price Prediction Platform")
if model is not None:
    st.success(f"สถานะโมเดล: `{model_status}` (พร้อมใช้งาน 51 Features)")
else:
    st.error(f"สถานะโมเดล: `{model_status}`")

col_input, col_result = st.columns([1.1, 0.9], gap="large")

with col_input:
    st.subheader("📋 ระบุคุณลักษณะของบ้าน")
    
    with st.expander("1. โซนและทำเล (Zoning & Lot Configuration)", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            ms_zoning = st.selectbox(
                "โซนผังเมือง (MSZoning)",
                ["RL (Low Density)", "RM (Medium Density)", "FV (Floating Village)", "RH (High Density)"]
            )
            lot_shape = st.selectbox(
                "รูปทรงที่ดิน (LotShape)",
                ["Reg (สี่เหลี่ยมปกติ)", "IR1 (ไม่ปกติเล็กน้อย)", "IR2 (ค่อนข้างเบี้ยว)", "IR3 (ไม่ปกติมาก)"]
            )
        with c2:
            lot_config = st.selectbox(
                "ลักษณะแปลงที่ดิน (LotConfig)",
                ["Inside (แปลงใน)", "Corner (แปลงหัวมุม)", "CulDSac (ซอยตัน)", "FR2 (ติดถนน 2 ด้าน)", "FR3 (ติดถนน 3 ด้าน)"]
            )
            neighborhood = st.slider("ทำเล / โซนเพื่อนบ้าน (Neighborhood Code: 0-25)", min_value=0, max_value=25, value=12)

    with st.expander("2. โครงสร้าง สไตล์ และวัสดุ (Style & Foundation)", expanded=True):
        c3, c4 = st.columns(2)
        with c3:
            house_style = st.selectbox(
                "สไตล์ตัวบ้าน (HouseStyle)",
                ["1Story (ชั้นเดียว)", "2Story (2 ชั้น)", "1.5Fin (ชั้นครึ่งเสร็จ)", "1.5Unf (ชั้นครึ่งไม่เสร็จ)", 
                 "2.5Fin", "2.5Unf", "SFoyer", "SLvl (เล่นระดับ)"]
            )
            roof_style = st.selectbox(
                "ทรงหลังคา (RoofStyle)",
                ["Gable (จั่ว)", "Hip (ปั้นหยา)", "Gambrel", "Mansard", "Flat", "Shed"]
            )
        with c4:
            foundation = st.selectbox(
                "โครงสร้างฐานราก (Foundation)",
                ["PConc (คอนกรีตเท)", "CBlock (คอนกรีตบล็อก)", "BrkTil (อิฐ)", "Slab", "Stone", "Wood"]
            )
            mas_vnr = st.selectbox(
                "การตกแต่งผนังภายนอก (Masonry Veneer)",
                ["None (ไม่มี)", "BrkFace (ก่ออิฐโชว์แนว)", "Stone (หิน)", "BrkCmn"]
            )

    with st.expander("3. โรงจอดรถและคุณภาพภายใน (Quality & Garage)", expanded=True):
        qual_mapping = {
            "ยอดเยี่ยม (Excellent)": 5.0,
            "ดีมาก (Good)": 4.0,
            "มาตรฐานทั่วไป (Typical/Average)": 3.0,
            "พอใช้ (Fair)": 2.0,
            "แย่ / ต้องปรับปรุง (Poor)": 1.0
        }
        
        c5, c6 = st.columns(2)
        with c5:
            garage_type = st.selectbox(
                "ประเภทโรงจอดรถ (GarageType)",
                ["Attchd (ติดกับตัวบ้าน)", "Detchd (แยกจากบ้าน)", "BuiltIn (ในตัวบ้าน)", "Basment", "CarPort", "NoGarage (ไม่มี)"]
            )
            garage_finish_choice = st.selectbox(
                "การตกแต่งโรงจอดรถ (Garage Finish)",
                ["ตกแต่งสมบูรณ์ (Finished)", "ตกแต่งบางส่วน (Rough Finished)", "ยังไม่ตกแต่ง (Unfinished)", "ไม่มีโรงรถ (No Garage)"],
                index=1
            )
            kitchen_choice = st.selectbox(
                "คุณภาพห้องครัว (Kitchen Quality)",
                options=list(qual_mapping.keys()),
                index=2
            )
        with c6:
            heating_choice = st.selectbox(
                "ระบบทำความร้อน (Heating Quality)",
                options=list(qual_mapping.keys()),
                index=1
            )
            has_bsmt = st.checkbox("มีห้องใต้ดิน (Basement)", value=True)
            bsmt_choice = st.selectbox(
                "เกรดห้องใต้ดิน (Basement Quality)",
                options=list(qual_mapping.keys()),
                index=2,
                disabled=not has_bsmt
            )

    btn_predict = st.button("🔮 คำนวณราคาประเมิน (Predict)", type="primary", use_container_width=True)

with col_result:
    st.subheader("📊 ผลการประเมินราคา")
    
    if btn_predict:
        if model is None:
            st.error("ไม่สามารถคำนวณได้เนื่องจากโมเดลยังไม่พร้อมใช้งาน")
        else:
            row_data = {col: 0.0 for col in FEATURE_NAMES}
            
            # One-Hot Encoding
            if "FV" in ms_zoning: row_data['MSZoning_FV'] = 1.0
            elif "RH" in ms_zoning: row_data['MSZoning_RH'] = 1.0
            elif "RL" in ms_zoning: row_data['MSZoning_RL'] = 1.0
            elif "RM" in ms_zoning: row_data['MSZoning_RM'] = 1.0

            if "IR2" in lot_shape: row_data['LotShape_IR2'] = 1.0
            elif "IR3" in lot_shape: row_data['LotShape_IR3'] = 1.0
            elif "Reg" in lot_shape: row_data['LotShape_Reg'] = 1.0

            if "CulDSac" in lot_config: row_data['LotConfig_CulDSac'] = 1.0
            elif "FR2" in lot_config: row_data['LotConfig_FR2'] = 1.0
            elif "FR3" in lot_config: row_data['LotConfig_FR3'] = 1.0
            elif "Inside" in lot_config: row_data['LotConfig_Inside'] = 1.0

            style_map = {
                "1.5Unf": 'HouseStyle_1.5Unf', "1Story": 'HouseStyle_1Story',
                "2.5Fin": 'HouseStyle_2.5Fin', "2.5Unf": 'HouseStyle_2.5Unf',
                "2Story": 'HouseStyle_2Story', "SFoyer": 'HouseStyle_SFoyer', "SLvl": 'HouseStyle_SLvl'
            }
            for k, v in style_map.items():
                if k in house_style:
                    row_data[v] = 1.0

            roof_map = {
                "Gable": 'RoofStyle_Gable', "Gambrel": 'RoofStyle_Gambrel',
                "Hip": 'RoofStyle_Hip', "Mansard": 'RoofStyle_Mansard', "Shed": 'RoofStyle_Shed'
            }
            for k, v in roof_map.items():
                if k in roof_style:
                    row_data[v] = 1.0

            if "BrkFace" in mas_vnr: row_data['MasVnrType_BrkFace'] = 1.0
            elif "None" in mas_vnr: row_data['MasVnrType_None'] = 1.0
            elif "Stone" in mas_vnr: row_data['MasVnrType_Stone'] = 1.0

            # Default ExterQual Dummy
            row_data['ExterQual_TA'] = 1.0

            found_map = {
                "CBlock": 'Foundation_CBlock', "PConc": 'Foundation_PConc',
                "Slab": 'Foundation_Slab', "Stone": 'Foundation_Stone', "Wood": 'Foundation_Wood'
            }
            for k, v in found_map.items():
                if k in foundation:
                    row_data[v] = 1.0

            garage_map = {
                "Attchd": 'GarageType_Attchd', "Basment": 'GarageType_Basment',
                "BuiltIn": 'GarageType_BuiltIn', "CarPort": 'GarageType_CarPort',
                "Detchd": 'GarageType_Detchd', "NoGarage": 'GarageType_NoGarage'
            }
            for k, v in garage_map.items():
                if k in garage_type:
                    row_data[v] = 1.0
            
            garage_finish_map = {
                "ตกแต่งสมบูรณ์ (Finished)": 3.0,
                "ตกแต่งบางส่วน (Rough Finished)": 2.0,
                "ยังไม่ตกแต่ง (Unfinished)": 1.0,
                "ไม่มีโรงรถ (No Garage)": 0.0
            }

            # ตัวแปรเชิงตัวเลขและ Ordinal Features
            row_data['Neighborhood'] = float(neighborhood)
            row_data['Exterior1st'] = 12.0
            row_data['Exterior2nd'] = 13.0
            row_data['ExterQual'] = 3.0
            row_data['BsmtQual'] = qual_mapping[bsmt_choice] if has_bsmt else 0.0
            row_data['BsmtExposure'] = 1.0
            row_data['BsmtFinType1'] = 4.0
            row_data['HeatingQC'] = qual_mapping[heating_choice]
            row_data['KitchenQual'] = qual_mapping[kitchen_choice]
            row_data['FireplaceQu'] = 3.0
            row_data['GarageFinish'] = garage_finish_map[garage_finish_choice]

            input_df = pd.DataFrame([row_data])[FEATURE_NAMES]

            try:
                pred = model.predict(input_df)
                price = float(pred[0])

                if price >= 0:
                    st.metric(
                        label="ราคาประเมินจาก Linear Regression Model",
                        value=f"${price:,.2f}"
                    )
                else:
                    st.metric(
                        label="ราคาประเมินจาก Linear Regression Model",
                        value=f"-${abs(price):,.2f}"
                    )
                    st.warning("⚠️ ผลการคำนวณติดลบ: เนื่องจากค่าสัมประสิทธิ์ตัวแปรคุณภาพต่ำกว่าค่า Intercept แนะนำให้ปรับเกรดห้องครัว, ระบบทำความร้อน หรือโรงรถให้สูงขึ้น")

                with st.expander("รายละเอียดโมเดล (Model Internals)"):
                    if hasattr(model, "intercept_"):
                        st.write(f"**Base Intercept:** `${float(model.intercept_):,.2f}`")
                    st.write("**ตาราง Feature Vector (51 Features) ที่ส่งเข้าโมเดล:**")
                    st.dataframe(input_df)

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการ Predict: {e}")
    else:
        st.info("👈 ปรับแต่งตัวแปรทางด้านซ้าย แล้วกดปุ่ม **'คำนวณราคาประเมิน (Predict)'** เพื่อให้โมเดลเริ่มทำงาน")
