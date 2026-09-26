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
MODEL_PATH = "linear_regression_model1.pkl" 

# รายชื่อ 76 Features ตามลำดับที่โมเดลต้องการ
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

NEIGHBORHOOD_MAP = {'CollgCr': np.float64(12.169332259956139), 
             'Veenker': np.float64(12.266186946338435), 
             'Crawfor': np.float64(12.237256187249674), 
             'NoRidge': np.float64(12.596944834295265),
             'Mitchel': np.float64(11.967956343260019), 
             'Somerst': np.float64(12.29107485188606), 
             'NWAmes': np.float64(12.137222248623926), 
             'OldTown': np.float64(11.73049818713178), 
             'BrkSide': np.float64(11.668680867662754), 
             'Sawyer': np.float64(11.805959895258422), 
             'NridgHt': np.float64(12.589057073179353), 
             'NAmes': np.float64(11.877717162561297), 
             'SawyerW': np.float64(12.040539413569535), 
             'IDOTRR': np.float64(11.563381912602678), 
             'MeadowV': np.float64(11.557207645181311), 
             'Edwards': np.float64(11.745846688038602), 
             'Timber': np.float64(12.32619253054482), 
             'Gilbert': np.float64(12.171106002634197), 
             'StoneBr': np.float64(12.524199680812767), 
             'ClearCr': np.float64(12.279718567435713),
             'NPkVill': np.float64(11.884201006921435), 
             'Blmngtn': np.float64(12.14777689419568),
             'BrDale': np.float64(11.540679043361267),
             'SWISU': np.float64(11.874302215307473), 
             'Blueste': np.float64(12.029983731526684)}

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"ไม่พบไฟล์ '{MODEL_PATH}'"
    try:
        loaded_model = joblib.load(MODEL_PATH)
        return loaded_model, "โหลดโมเดลสำเร็จ"
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
            selected_neighborhood = st.selectbox("ย่านที่ตั้งของบ้าน (Neighborhood)", options=list(NEIGHBORHOOD_MAP.keys()))
            ms_zoning_th = st.selectbox(
                "โซนผังเมือง (MSZoning)", 
                [
                    "ที่อยู่อาศัยหนาแน่นต่ำ (บ้านเดี่ยวทั่วไป)", 
                    "ที่อยู่อาศัยหนาแน่นปานกลาง (เช่น ทาวน์เฮาส์ ตึกแถว)", 
                    "โครงการหมู่บ้านจัดสรรริมน้ำ/สไตล์วิลเลจ", 
                    "ที่อยู่อาศัยหนาแน่นสูง (เช่น คอนโด อพาร์ตเมนต์สูง)"
                ]
            )

    with st.expander("4. เกรดและคุณภาพวัสดุ (Quality Ratings)", expanded=True):
        qual_options = {"Excellent": 5.0, "Good": 4.0, "Typical": 3.0, "Fair": 2.0, "Poor": 1.0}
        c9, c10 = st.columns(2)
        with c9:
            kitchen_qual = st.selectbox("คุณภาพห้องครัว", list(qual_options.keys()), index=2)
            exter_qual = st.selectbox("คุณภาพวัสดุภายนอก", list(qual_options.keys()), index=2)
        with c10:
            heating_qc = st.selectbox("คุณภาพระบบทำความร้อน", list(qual_options.keys()), index=0)
            bsmt_qual = st.selectbox("คุณภาพห้องใต้ดิน", list(qual_options.keys()), index=2)

    btn_predict = st.button("🔮 คำนวณราคาประเมิน (Predict)", type="primary", use_container_width=True)

with col_result:
    st.subheader("📊 ผลการประเมินราคา")
    
    if btn_predict:
        if model is None:
            st.error("โมเดลไม่พร้อมใช้งาน กรุณาตรวจสอบไฟล์ .pkl")
        else:
            # 1. ดึงรายชื่อ features ที่แท้จริงจากตัวโมเดลโดยตรง
            if hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)
            else:
                expected_features = FEATURE_NAMES

            # 2. สร้างพจนานุกรมเก็บค่าเริ่มต้น 0.0 สำหรับทุกคอลัมน์ที่โมเดลต้องการ
            row_data = {col: 0.0 for col in expected_features}

            # แมปตัวเลือกภาษาไทยกลับเป็นรหัสย่อของชุดข้อมูล
            zoning_map = {
                "ที่อยู่อาศัยหนาแน่นต่ำ (บ้านเดี่ยวทั่วไป)": "RL",
                "ที่อยู่อาศัยหนาแน่นปานกลาง (เช่น ทาวน์เฮาส์ ตึกแถว)": "RM",
                "โครงการหมู่บ้านจัดสรรริมน้ำ/สไตล์วิลเลจ": "FV",
                "ที่อยู่อาศัยหนาแน่นสูง (เช่น คอนโด อพาร์ตเมนต์สูง)": "RH"
            }
            selected_zoning = zoning_map[ms_zoning_th]
            

            # 3. แมปตัวแปรหลักที่มีใน UI เข้ากับชื่อคอลัมน์
            value_map = {
                'GrLivArea': float(gr_liv_area),
                'LotArea': float(lot_area),
                '1stFlrSF': float(first_flr_sf),
                '2ndFlrSF': float(second_flr_sf),
                'TotalBsmtSF': float(total_bsmt_sf),
                'GarageArea': float(garage_area),
                'BedroomAbvGr': float(bedroom),
                'TotRmsAbvGrd': float(tot_rms),
                'FullBath': float(full_bath),
                'GarageCars': float(garage_cars),
                'HouseAge': float(house_age),
                'RemodAge': float(remod_age) if is_remodeled else float(house_age),
                'IsRemodeled': 1.0 if is_remodeled else 0.0,
                'HalfBath_binned': 1.0 if half_bath else 0.0,
                'Fireplaces_binned': 1.0 if has_fireplace else 0.0,
                'WoodDeckSF_binary': 1.0 if has_wood_deck else 0.0,
                'MSSubClass': 20.0,
                'LotFrontage': 70.0,
                'MoSold': 6.0,
                f'MSZoning_{selected_zoning}': 1.0,
                'LotShape_Reg': 1.0,
                'LotConfig_Inside': 1.0,
                'HouseStyle_1Story': 1.0 if second_flr_sf == 0 else 0.0,
                'HouseStyle_2Story': 1.0 if second_flr_sf > 0 else 0.0,
                'RoofStyle_Gable': 1.0,
                'Foundation_PConc': 1.0,
                'GarageType_Attchd': 1.0,
                'MasVnrType_None': 1.0,
                'Neighborhood': NEIGHBORHOOD_MAP[selected_neighborhood],
                'Exterior1st': 12.0,
                'Exterior2nd': 13.0,
                'KitchenQual': qual_options[kitchen_qual],
                'ExterQual': qual_options[exter_qual],
                'HeatingQC': qual_options[heating_qc],
                'BsmtQual': qual_options[bsmt_qual] if total_bsmt_sf > 0 else 0.0,
                'FireplaceQu': 3.0 if has_fireplace else 0.0,
                'GarageFinish': 2.0 if garage_cars > 0 else 0.0,
                'BsmtExposure': 1.0,
                'BsmtFinType1': 4.0
            }
                
            for feature, val in value_map.items():
                if feature in row_data:
                    row_data[feature] = val

            # 4. แปลงเป็น DataFrame โดยใช้ลำดับคอลัมน์ของโมเดล 100%
            input_df = pd.DataFrame([row_data])[expected_features]

            try:
                raw_pred = float(model.predict(input_df)[0])
                
                # --- จุดแปลงค่า Log กลับเป็น Dollar จริง ---
                # หากค่า raw_pred มีค่าน้อย (ช่วง Log สเกล 5 - 25) ให้แปลงกลับด้วย expm1
                if 0 < raw_pred < 30:
                    real_price = np.expm1(raw_pred)
                elif raw_pred <= 0:
                    real_price = 0.0
                else:
                    real_price = raw_pred

                st.metric(
                    label="ราคาประเมินจริง (Estimated Sale Price)", 
                    value=f"${real_price:,.2f}"
                )
                st.caption(f"ค่าดิบที่ได้จากโมเดล (Log Scale Output): `{raw_pred:.4f}`")

                if real_price <= 0:
                    st.warning("⚠️ ผลลัพธ์ผิดปกติ: ลองปรับพื้นที่ใช้สอยให้มากขึ้น หรือลดอายุของบ้านลง")

                with st.expander("ดูตาราง Features (ที่ส่งเข้าโมเดล)"):
                    st.dataframe(input_df.T, height=400)

            except Exception as e:
                st.error(f"Prediction Error: {e}")
    else:
        st.info("👈 ระบุรายละเอียดพื้นที่บ้านทางด้านซ้าย แล้วกดปุ่มเพื่อเริ่มประเมินราคา")
