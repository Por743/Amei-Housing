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

EXTERIOR1ST_MAP = {'VinylSd': np.float64(12.189920420399533), 'MetalSd': np.float64(11.874175192515807), 'Wd Sdng': np.float64(11.835468282628185), 'HdBoard': np.float64(11.944799536084124), 'BrkFace': np.float64(12.218156276546033), 'WdShing': np.float64(11.702559919011081), 'CemntBd': np.float64(12.284045439327102), 'Plywood': np.float64(12.053988401909287), 'AsbShng': np.float64(11.672263645974672), 'Stucco': np.float64(12.089706611689746), 'BrkComm': np.float64(11.224442501841812), 'AsphShn': np.float64(12.029983731526684), 'Stone': np.float64(12.345838935721968), 'ImStucc': np.float64(12.476103599529843), 'CBlock': np.float64(11.561725152903833), 'None': 12.029983731526684}
EXTERIOR2ND_MAP = {'VinylSd': np.float64(12.193646655471017), 'MetalSd': np.float64(11.881719342949529), 'Wd Shng': np.float64(11.845161658207267), 'HdBoard': np.float64(11.970826237484328), 'Plywood': np.float64(11.992985891366112), 'Wd Sdng': np.float64(11.83768699539125), 'CmentBd': np.float64(12.281718483455007), 'BrkFace': np.float64(12.356996718765055), 'Stucco': np.float64(12.017083586945958), 'AsbShng': np.float64(11.727787402555176), 'Brk Cmn': np.float64(11.712311235085469), 'ImStucc': np.float64(12.226166858483968), 'AsphShn': np.float64(11.96049335528058), 'Stone': np.float64(12.20478346531541), 'CBlock': np.float64(11.561725152903833), 'None': 12.029983731526684}

MSSUBCLASS_MAP = {
    '1-STORY 1946 & NEWER': np.float64(12.064447214828162),
    '1-STORY 1945 & OLDER': np.float64(11.445593094097848),
    '1-STORY FINISHED ATTIC': np.float64(12.060346943152025),
    '1-1/2 STORY UNFINISHED ATTIC': np.float64(11.560327154371183), 
    '1-1/2 STORY FINISHED ATTIC': np.float64(11.78915908021649),
    '2-STORY 1946 & NEWER': np.float64(12.319992050615664), 
    '2-STORY 1945 & OLDER': np.float64(12.001508611509148), 
    '2-1/2 STORY ALL AGES': np.float64(12.104289369619238),
    'SPLIT OR MULTI-LEVEL': np.float64(12.005705580571783),
    'SPLIT FOYER': np.float64(11.912601668656484),
    'DUPLEX': np.float64(11.817736089772582),
    '1-STORY PUD 1946 & NEWER': np.float64(12.186210887326258),
    '2-STORY PUD - 1946 & NEWER': np.float64(11.810688677393426), 
    'PUD MULTILEVEL INCL SPLIT LEV/FOYER': np.float64(11.745116592703285),
    '2 FAMILY CONVERSION': np.float64(11.73539691300343)
}

# --- พจนานุกรมสำหรับแมป MSSubClass ไปเป็น HouseStyle อัตโนมัติ ---
SUBCLASS_TO_HOUSESTYLE = {
    '1-STORY 1946 & NEWER': 'HouseStyle_1Story',
    '1-STORY 1945 & OLDER': 'HouseStyle_1Story',
    '1-STORY FINISHED ATTIC': 'HouseStyle_1Story',
    '1-1/2 STORY UNFINISHED ATTIC': 'HouseStyle_1.5Unf', 
    '1-1/2 STORY FINISHED ATTIC': 'HouseStyle_1.5Fin',
    '2-STORY 1946 & NEWER': 'HouseStyle_2Story', 
    '2-STORY 1945 & OLDER': 'HouseStyle_2Story', 
    '2-1/2 STORY ALL AGES': 'HouseStyle_2.5Unf',
    'SPLIT OR MULTI-LEVEL': 'HouseStyle_SLvl',
    'SPLIT FOYER': 'HouseStyle_SFoyer',
    'DUPLEX': 'HouseStyle_2Story',
    '1-STORY PUD 1946 & NEWER': 'HouseStyle_1Story',
    '2-STORY PUD - 1946 & NEWER': 'HouseStyle_2Story', 
    'PUD MULTILEVEL INCL SPLIT LEV/FOYER': 'HouseStyle_SLvl',
    '2 FAMILY CONVERSION': 'HouseStyle_2Story'
}

shape_map = {
    "ที่ดินรูปทรงสี่เหลี่ยมปกติ (Regular)": "Reg",
    "ที่ดินรูปทรงเบี้ยว (Moderately Irregular)": "IR2",
    "ที่ดินรูปทรงอิสระ (Irregular - IR3)": "IR3"
}

roof_map = {
    "หลังคาหน้าจั่ว (Gable - มาตรฐาน)": "Gable",
    "หลังคาทรงปั้นหยา (Hip)": "Hip",
    "หลังคาทรงแบน (Flat)": "Flat",  # ตัวนี้เป็น Baseline ไม่มีในคอลัมน์โมเดล
    "หลังคาทรงแกมเบรล/ยุ้งฉาง (Gambrel)": "Gambrel",
    "หลังคาทรงมังซาร์ (Mansard)": "Mansard",
    "หลังคาทรงเพิงหมาแหงน (Shed)": "Shed"
}
foundation_map = {
    "คอนกรีตเทสำเร็จ / คอนกรีตหล่อ (Poured Concrete - PConc)": "PConc",
    "บล็อกคอนกรีตอัดแรง (Cinder Block - CBlock)": "CBlock",
    "พื้นคอนกรีตวางบนคานดิน (Slab)": "Slab",
    "ฐานรากหินธรรมชาติ (Stone)": "Stone",
    "ฐานรากโครงสร้างไม้ (Wood)": "Wood"
}

garage_type_map = {
    "โรงรถติดกับตัวบ้าน (Attached - Attchd)": "Attchd",
    "โรงรถแยกจากตัวบ้าน (Detached - Detchd)": "Detchd",
    "โรงรถฝังในตัวบ้าน/ใต้ห้องชั้นสอง (Built-In)": "BuiltIn",
    "โรงรถชั้นใต้ดิน (Basement - Basment)": "Basment",
    "โรงจอดรถแบบหลังคาโปร่ง/เพิงจอดรถ (CarPort)": "CarPort",
    "ไม่มีโรงจอดรถ (No Garage)": "NoGarage",
    "โรงจอดรถมากกว่าหนึ่งรูปแบบ (More than one type - 2Types)": "2Types"  # Baseline (ไม่มีในคอลัมน์โมเดล)
}

mas_vnr_map = {
    "ไม่มีการกรุอิฐ/หินประดับ (None)": "None",
    "กรุอิฐโชว์แนวเกรดดี (Brick Face - BrkFace)": "BrkFace",
    "กรุหินธรรมชาติ (Stone)": "Stone",
    "กรุอิฐมอญธรรมดา (Brick Common - BrkCmn)": "BrkCmn"  # Baseline (ไม่มีในคอลัมน์โมเดล)
}

bsmt_exposure_options = {
    "ทึบแสง มิดชิดใต้ดิน (No Exposure)": 1.0,
    "แสงส่องถึงเล็กน้อย (Minimum Exposure)": 2.0,
    "แสงส่องถึงปานกลาง (Average Exposure)": 3.0,
    "แสงส่องถึงดีมาก / มีทางเดินออกระดับดิน (Good Exposure)": 4.0
}

bsmt_fintype_options = {
    "ยังไม่ตกแต่ง เป็นปูนเปลือย (Unfinished - Unf)": 1.0,
    "ตกแต่งระดับพื้นฐาน (Low Quality - LwQ)": 2.0,
    "ตกแต่งเป็นห้องสันทนาการ (Rec Room)": 3.0,
    "ตกแต่งอยู่อาศัยระดับทั่วไป (Below Average - BLQ)": 4.0,
    "ตกแต่งอยู่อาศัยระดับดี (Average Living - ALQ)": 5.0,
    "ตกแต่งอยู่อาศัยสมบูรณ์แบบ/เกรดพรีเมียม (Good Living - GLQ)": 6.0
}

# กำหนดค่าเริ่มต้นให้กับ session_state เพื่อป้องกันข้อมูลหายเมื่อมีการ rerun
if "predicted_price" not in st.session_state:
    st.session_state["predicted_price"] = None
    st.session_state["raw_pred"] = None
    st.session_state["input_df_display"] = None

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
            lot_frontage = st.number_input("ความกว้างหน้าที่ดินติดถนน (ฟุต - LotFrontage)", min_value=10, max_value=400, value=70, step=5)
            lot_shape = st.selectbox("รูปทรงของแปลงที่ดิน (Lot Shape)",
                                         options=list(shape_map.keys()), index=0
                                        )
            selected_subclass = st.selectbox("ประเภทของบ้าน", options=list(MSSUBCLASS_MAP.keys()))
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
            garage_type_th = st.selectbox(
                "ประเภทโรงจอดรถ (Garage Type)", 
                options=list(garage_type_map.keys()), 
                index=0 if garage_cars > 0 else 5)
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
            foundation_th = st.selectbox("ประเภทฐานราก (Foundation)", options=list(foundation_map.keys()), index=0)
            roof_style_th = st.selectbox("รูปทรงหลังคา (Roof Style)", options=list(roof_map.keys()), index=0)
            selected_ext1 = st.selectbox("วัสดุภายนอก (Exterior 1st)",
                                         options=list(EXTERIOR1ST_MAP.keys()), index=15)
            selected_ext2 = st.selectbox("วัสดุภายนอก(หากมีหลายวัสดุ) (Exterior 2nd)",
                                        options=list(EXTERIOR2ND_MAP.keys()), index=15)
            mas_vnr_th = st.selectbox("วัสดุตกแต่งผนังภายนอก (Masonry Veneer)", options=list(mas_vnr_map.keys()), index=0)
        with c10:
            exter_qual = st.selectbox("คุณภาพวัสดุ", list(qual_options.keys()), index=2)
            kitchen_qual = st.selectbox("คุณภาพห้องครัว", list(qual_options.keys()), index=2)
            heating_qc = st.selectbox("คุณภาพระบบทำความร้อน", list(qual_options.keys()), index=2)
            has_bsmt = total_bsmt_sf > 0
            bsmt_qual = st.selectbox("คุณภาพโครงสร้างห้องใต้ดิน", list(qual_options.keys()), index=2, disabled=not has_bsmt)
            bsmt_exposure_th = st.selectbox(
                "การเปิดรับแสงของห้องใต้ดิน (BsmtExposure)", 
                options=list(bsmt_exposure_options.keys()), 
                index=0, 
                disabled=not has_bsmt
            )
            bsmt_fin_type_th = st.selectbox(
                "ระดับการตกแต่งห้องใต้ดิน (BsmtFinType1)", 
                options=list(bsmt_fintype_options.keys()), 
                index=0, 
                disabled=not has_bsmt
            )
            
            btn_predict = st.button("🔮 คำนวณราคาประเมิน (Predict)", type="primary", use_container_width=True)

with col_result:
    st.subheader("📊 ผลการประเมินราคา")
    
    # 1. จัดการลอจิกการกดปุ่มเพื่อคำนวณและบันทึกผลลง st.session_state
    if btn_predict:
        if model is None:
            st.error("โมเดลไม่พร้อมใช้งาน กรุณาตรวจสอบไฟล์ .pkl")
        else:
            if hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)
            else:
                expected_features = FEATURE_NAMES

            row_data = {col: 0.0 for col in expected_features}

            zoning_map = {
                "ที่อยู่อาศัยหนาแน่นต่ำ (บ้านเดี่ยวทั่วไป)": "RL",
                "ที่อยู่อาศัยหนาแน่นปานกลาง (เช่น ทาวน์เฮาส์ ตึกแถว)": "RM",
                "โครงการหมู่บ้านจัดสรรริมน้ำ/สไตล์วิลเลจ": "FV",
                "ที่อยู่อาศัยหนาแน่นสูง (เช่น คอนโด อพาร์ตเมนต์สูง)": "RH"
            }
            
            # ดึงรหัสย่อของ LotShape จาก dictionary
            selected_shape_code = shape_map[lot_shape]
            selected_roof_code = roof_map[roof_style_th]
            selected_zoning = zoning_map[ms_zoning_th]
            selected_foundation_code = foundation_map[foundation_th]
            selected_garage_code = garage_type_map[garage_type_th]
            selected_mas_vnr_code = mas_vnr_map[mas_vnr_th]
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
                'MSSubClass': MSSUBCLASS_MAP[selected_subclass],
                'LotFrontage': float(lot_frontage),
                f'MSZoning_{selected_zoning}': 1.0,
                f'LotShape_{selected_shape_code}': 1.0,  # แก้ไขบั๊กใช้ชื่อย่อตรงนี้
                'LotConfig_Inside': 1.0,
                f'RoofStyle_{selected_roof_code}': 1.0,
                f'Foundation_{selected_foundation_code}': 1.0,
                f'GarageType_{selected_garage_code}': 1.0,
                f'MasVnrType_{selected_mas_vnr_code}': 1.0,
                'Neighborhood': NEIGHBORHOOD_MAP[selected_neighborhood],
                'Exterior1st': EXTERIOR1ST_MAP[selected_ext1],
                'Exterior2nd': EXTERIOR2ND_MAP[selected_ext2],
                'KitchenQual': qual_options[kitchen_qual],
                'ExterQual': qual_options[exter_qual],
                'HeatingQC': qual_options[heating_qc],
                'BsmtQual': qual_options[bsmt_qual] if total_bsmt_sf > 0 else 0.0,
                'FireplaceQu': 3.0 if has_fireplace else 0.0,
                'GarageFinish': 2.0 if garage_cars > 0 else 0.0,
                'BsmtExposure': bsmt_exposure_options[bsmt_exposure_th] if total_bsmt_sf > 0 else 0.0,
                'BsmtFinType1': bsmt_fintype_options[bsmt_fintype_th] if total_bsmt_sf > 0 else 0.0,
            }
                
            # --- อัปเดตตัวแปรทั่วไปเข้า row_data ---
            for feature, val in value_map.items():
                if feature in row_data:
                    row_data[feature] = val

            # --- แมป HouseStyle อัตโนมัติจาก MSSubClass ---
            # ดึงชื่อคอลัมน์ HouseStyle ที่ตรงกับประเภทบ้านที่เลือก
            target_style_col = SUBCLASS_TO_HOUSESTYLE.get(selected_subclass, None)

            # ถ้าคอลัมน์นั้นมีอยู่ใน 76 ฟีเจอร์ของโมเดล ให้ตั้งค่าเป็น 1.0
            if target_style_col and target_style_col in row_data:
                row_data[target_style_col] = 1.0

            input_df = pd.DataFrame([row_data])[expected_features]

            try:
                raw_pred = float(model.predict(input_df)[0])
                
                if 0 < raw_pred < 30:
                    real_price = np.expm1(raw_pred)
                elif raw_pred <= 0:
                    real_price = 0.0
                else:
                    real_price = raw_pred

                # บันทึกค่าลงใน Session State แทนการแสดงผลทันที
                st.session_state["predicted_price"] = real_price
                st.session_state["raw_pred"] = raw_pred
                st.session_state["input_df_display"] = input_df

            except Exception as e:
                st.error(f"Prediction Error: {e}")

    # 2. จัดการลอจิกการแสดงผล (ดึงข้อมูลจาก st.session_state มาแสดงเสมอถ้ามีข้อมูล)
    if st.session_state["predicted_price"] is not None:
        real_price = st.session_state["predicted_price"]
        raw_pred = st.session_state["raw_pred"]
        input_df = st.session_state["input_df_display"]

        st.metric(
            label="ราคาประเมินจริง (Estimated Sale Price)", 
            value=f"${real_price:,.2f}"
        )
        st.caption(f"ค่าดิบที่ได้จากโมเดล (Log Scale Output): `{raw_pred:.4f}`")

        if real_price <= 0:
            st.warning("⚠️ ผลลัพธ์ผิดปกติ: ลองปรับพื้นที่ใช้สอยให้มากขึ้น หรือลดอายุของบ้านลง")

        with st.expander("ดูตาราง Features (ที่ส่งเข้าโมเดล)"):
            st.dataframe(input_df.T, height=400)
    else:
        st.info("👈 ระบุรายละเอียดพื้นที่บ้านทางด้านซ้าย แล้วกดปุ่มเพื่อเริ่มประเมินราคา")
