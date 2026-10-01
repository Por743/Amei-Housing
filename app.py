import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="House Price Valuation",
    page_icon="🏡",
    layout="wide"
)

# ตั้งชื่อไฟล์โมเดลให้ตรงกับที่คุณอัปโหลด
MODEL_PATH = "linear_regression_model.pkl" 

# รายชื่อ Features ใหม่ที่อัปเดต (87 คอลัมน์ ตัด MSSubClass ที่ซ้ำออก)
FEATURE_NAMES = [
    'LotFrontage', 'LotArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF',
    '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr',
    'TotRmsAbvGrd', 'GarageYrBlt', 'GarageCars', 'GarageArea',
    'KitchenAbvGr_binned', 'BsmtHalfBath_binned', 'HalfBath_binned',
    'BsmtFullBath_binned', 'Fireplaces_binned', 'OverallCond_binned',
    'OverallQual_binned', 'TotalPorchSF', 'ScreenPorch_binary',
    'EnclosedPorch_binary', 'BsmtFinSF2_binary', 'MasVnrArea_binary',
    'OpenPorchSF_binary', '3SsnPorch_binary', 'WoodDeckSF_binary',
    'LowQualFinSF_binary', 'PoolArea_binary', 'MiscVal_binary', 'HouseAge',
    'RemodAge', 'IsRemodeled', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL',
    'MSZoning_RM', 'LotShape_IR2', 'LotShape_IR3', 'LotShape_Reg',
    'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_FR3',
    'LotConfig_Inside', 'HouseStyle_1.5Unf', 'HouseStyle_1Story',
    'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_2Story',
    'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'RoofStyle_Gable',
    'RoofStyle_Gambrel', 'RoofStyle_Hip', 'RoofStyle_Mansard',
    'RoofStyle_Shed', 'MasVnrType_BrkFace', 'MasVnrType_None',
    'MasVnrType_Stone', 'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA',
    'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab',
    'Foundation_Stone', 'Foundation_Wood', 'GarageType_Attchd',
    'GarageType_Basment', 'GarageType_BuiltIn', 'GarageType_CarPort',
    'GarageType_Detchd', 'GarageType_NoGarage', 'Neighborhood',
    'Exterior1st', 'Exterior2nd', 'MSSubClass', 'ExterQual', 'BsmtQual',
    'BsmtExposure', 'BsmtFinType1', 'HeatingQC', 'KitchenQual',
    'FireplaceQu', 'GarageFinish'
]

NEIGHBORHOOD_MAP = {'CollgCr': np.float64(12.169332259956139), 'Veenker': np.float64(12.266186946338435), 'Crawfor': np.float64(12.237256187249674), 'NoRidge': np.float64(12.596944834295265), 'Mitchel': np.float64(11.967956343260019), 'Somerst': np.float64(12.29107485188606), 'NWAmes': np.float64(12.137222248623926), 'OldTown': np.float64(11.73049818713178), 'BrkSide': np.float64(11.668680867662754), 'Sawyer': np.float64(11.805959895258422), 'NridgHt': np.float64(12.589057073179353), 'NAmes': np.float64(11.877717162561297), 'SawyerW': np.float64(12.040539413569535), 'IDOTRR': np.float64(11.563381912602678), 'MeadowV': np.float64(11.557207645181311), 'Edwards': np.float64(11.745846688038602), 'Timber': np.float64(12.32619253054482), 'Gilbert': np.float64(12.171106002634197), 'StoneBr': np.float64(12.524199680812767), 'ClearCr': np.float64(12.279718567435713), 'NPkVill': np.float64(11.884201006921435), 'Blmngtn': np.float64(12.14777689419568), 'BrDale': np.float64(11.540679043361267), 'SWISU': np.float64(11.874302215307473), 'Blueste': np.float64(12.029983731526684)}
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

SUBCLASS_TO_HOUSESTYLE = {
    '1-STORY 1946 & NEWER': 'HouseStyle_1Story', '1-STORY 1945 & OLDER': 'HouseStyle_1Story', '1-STORY FINISHED ATTIC': 'HouseStyle_1Story',
    '1-1/2 STORY UNFINISHED ATTIC': 'HouseStyle_1.5Unf', '1-1/2 STORY FINISHED ATTIC': 'HouseStyle_1.5Fin',
    '2-STORY 1946 & NEWER': 'HouseStyle_2Story', '2-STORY 1945 & OLDER': 'HouseStyle_2Story', '2-1/2 STORY ALL AGES': 'HouseStyle_2.5Unf',
    'SPLIT OR MULTI-LEVEL': 'HouseStyle_SLvl', 'SPLIT FOYER': 'HouseStyle_SFoyer', 'DUPLEX': 'HouseStyle_2Story',
    '1-STORY PUD 1946 & NEWER': 'HouseStyle_1Story', '2-STORY PUD - 1946 & NEWER': 'HouseStyle_2Story', 
    'PUD MULTILEVEL INCL SPLIT LEV/FOYER': 'HouseStyle_SLvl', '2 FAMILY CONVERSION': 'HouseStyle_2Story'
}

# แมปค่าใหม่ ปรับให้ครอบคลุม Ordinal Encoding ตั้งแต่ระดับ 0 (ไม่มี) ถึงสูงสุด
shape_map = {"ที่ดินรูปทรงสี่เหลี่ยมปกติ (Regular)": "Reg", "ที่ดินรูปทรงเบี้ยว (Moderately Irregular)": "IR2", "ที่ดินรูปทรงอิสระ (Irregular - IR3)": "IR3"}
roof_map = {"หลังคาหน้าจั่ว (Gable)": "Gable", "หลังคาทรงปั้นหยา (Hip)": "Hip", "หลังคาทรงแบน (Flat)": "Flat", "หลังคาทรงแกมเบรล (Gambrel)": "Gambrel", "หลังคาทรงมังซาร์ (Mansard)": "Mansard", "หลังคาทรงเพิงหมาแหงน (Shed)": "Shed"}
foundation_map = {"คอนกรีตเทสำเร็จ (PConc)": "PConc", "บล็อกคอนกรีต (CBlock)": "CBlock", "พื้นคอนกรีตวางบนคานดิน (Slab)": "Slab", "ฐานรากหิน (Stone)": "Stone", "ฐานรากโครงสร้างไม้ (Wood)": "Wood"}
garage_type_map = {"โรงรถติดกับตัวบ้าน (Attchd)": "Attchd", "โรงรถแยกจากตัวบ้าน (Detchd)": "Detchd", "โรงรถฝังในตัวบ้าน (BuiltIn)": "BuiltIn", "โรงรถชั้นใต้ดิน (Basment)": "Basment", "เพิงจอดรถ (CarPort)": "CarPort", "ไม่มีโรงจอดรถ": "NoGarage"}
mas_vnr_map = {"ไม่มีการกรุประดับ (None)": "None", "กรุอิฐโชว์แนว (BrkFace)": "BrkFace", "กรุหินธรรมชาติ (Stone)": "Stone", "กรุอิฐมอญ (BrkCmn)": "BrkCmn"}

# --- ตารางปรับ Ordinal Mappings เพื่อให้ตรงสเกล 0-N (สำคัญมาก) ---
qual_options = {"ไม่มี (None)": 0.0, "แย่มาก (Poor)": 1.0, "พอใช้ (Fair)": 2.0, "ปานกลาง (Typical)": 3.0, "ดี (Good)": 4.0, "ดีเยี่ยม (Excellent)": 5.0}
bsmt_exposure_options = {"ไม่มีห้องใต้ดิน (None)": 0.0, "ทึบแสง (No Exposure)": 1.0, "แสงส่องถึงเล็กน้อย (Minimum)": 2.0, "แสงส่องถึงปานกลาง (Average)": 3.0, "แสงส่องถึงดีมาก (Good)": 4.0}
bsmt_fintype_options = {"ไม่มีห้องใต้ดิน (None)": 0.0, "เป็นปูนเปลือย (Unf)": 1.0, "ตกแต่งระดับพื้นฐาน (LwQ)": 2.0, "ห้องสันทนาการ (Rec)": 3.0, "ตกแต่งทั่วไป (BLQ)": 4.0, "ตกแต่งดี (ALQ)": 5.0, "เกรดพรีเมียม (GLQ)": 6.0}
garage_finish_options = {"ไม่มีโรงรถ (None)": 0.0, "ยังไม่ตกแต่ง (Unf)": 1.0, "ตกแต่งบางส่วน (RFn)": 2.0, "ตกแต่งสมบูรณ์ (Fin)": 3.0}

NEIGHBORHOOD_GEO = {
    'CollgCr': {'name': 'College Creek', 'lat': 42.0210, 'lon': -93.6850, 'radius': 600},
    'Veenker': {'name': 'Veenker', 'lat': 42.0405, 'lon': -93.6530, 'radius': 450},
    'Crawfor': {'name': 'Crawford', 'lat': 42.0150, 'lon': -93.6450, 'radius': 500},
    'NoRidge': {'name': 'Northridge', 'lat': 42.0505, 'lon': -93.6550, 'radius': 550},
    'Mitchel': {'name': 'Mitchell', 'lat': 41.9930, 'lon': -93.6050, 'radius': 600},
    'Somerst': {'name': 'Somerset', 'lat': 42.0520, 'lon': -93.6430, 'radius': 500},
    'NWAmes': {'name': 'Northwest Ames', 'lat': 42.0500, 'lon': -93.6330, 'radius': 700},
    'OldTown': {'name': 'Old Town', 'lat': 42.0290, 'lon': -93.6130, 'radius': 650},
    'BrkSide': {'name': 'Brookside', 'lat': 42.0260, 'lon': -93.6280, 'radius': 400},
    'Sawyer': {'name': 'Sawyer', 'lat': 42.0330, 'lon': -93.6680, 'radius': 600},
    'NridgHt': {'name': 'Northridge Heights', 'lat': 42.0600, 'lon': -93.6550, 'radius': 550},
    'NAmes': {'name': 'North Ames', 'lat': 42.0420, 'lon': -93.6200, 'radius': 800},
    'SawyerW': {'name': 'Sawyer West', 'lat': 42.0340, 'lon': -93.6850, 'radius': 500},
    'IDOTRR': {'name': 'Iowa DOT', 'lat': 42.0190, 'lon': -93.6230, 'radius': 450},
    'MeadowV': {'name': 'Meadow Village', 'lat': 41.9920, 'lon': -93.6120, 'radius': 350},
    'Edwards': {'name': 'Edwards', 'lat': 42.0220, 'lon': -93.6660, 'radius': 700},
    'Timber': {'name': 'Timberland', 'lat': 41.9980, 'lon': -93.6500, 'radius': 550},
    'Gilbert': {'name': 'Gilbert', 'lat': 42.0800, 'lon': -93.6480, 'radius': 650},
    'StoneBr': {'name': 'Stone Brook', 'lat': 42.0610, 'lon': -93.6330, 'radius': 450},
    'ClearCr': {'name': 'Clear Creek', 'lat': 42.0280, 'lon': -93.6520, 'radius': 500},
    'NPkVill': {'name': 'Northpark Villa', 'lat': 42.0500, 'lon': -93.6260, 'radius': 300},
    'Blmngtn': {'name': 'Bloomington Heights', 'lat': 42.0620, 'lon': -93.6420, 'radius': 400},
    'BrDale': {'name': 'Briardale', 'lat': 42.0525, 'lon': -93.6280, 'radius': 300},
    'SWISU': {'name': 'South & West of ISU', 'lat': 42.0180, 'lon': -93.6510, 'radius': 500},
    'Blueste': {'name': 'Bluestem', 'lat': 42.0090, 'lon': -93.6450, 'radius': 300}
}

qual_1_options = {
    "พอใช้ (Fair - Fa)": 0.0,
    "ปานกลาง (Typical - TA)": 1.0,
    "ดี (Good - Gd)": 2.0,
    "ดีเยี่ยม (Excellent - Ex)": 3.0
}
qual_2_options = {
    "แย่มาก (Poor - Po)": 0.0,
    "พอใช้ (Fair - Fa)": 1.0,
    "ปานกลาง (Typical - TA)": 2.0,
    "ดี (Good - Gd)": 3.0,
    "ดีเยี่ยม (Excellent - Ex)": 4.0
}

if "predicted_price" not in st.session_state:
    st.session_state["predicted_price"] = None
    st.session_state["raw_pred"] = None
    st.session_state["input_df_display"] = None



def render_neighborhood_map(selected_nh):
    geo_data = NEIGHBORHOOD_GEO.get(selected_nh, {'lat': 42.0308, 'lon': -93.6319, 'radius': 500, 'name': selected_nh})
    m = folium.Map(location=[geo_data['lat'], geo_data['lon']], zoom_start=14, tiles="OpenStreetMap")
    folium.Circle(
        location=[geo_data['lat'], geo_data['lon']], radius=geo_data['radius'],
        color="#2b7bba", weight=2, fill=True, fill_color="#3388ff", fill_opacity=0.35,
        tooltip=f"<b>{geo_data['name']} ({selected_nh})</b>"
    ).add_to(m)
    folium.Marker(
        location=[geo_data['lat'], geo_data['lon']], popup=f"📍 ย่าน: {geo_data['name']}",
        icon=folium.Icon(color="red", icon="home")
    ).add_to(m)
    return m
    
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH): return None, f"ไม่พบไฟล์ '{MODEL_PATH}'"
    try: return joblib.load(MODEL_PATH), "โหลดโมเดลสำเร็จ"
    except Exception as e: return None, f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}"

model, model_status = load_model()

st.title("🏡 Advanced House Price Prediction (Ames Housing)")
if model is not None:
    st.success(f"✅ สถานะโมเดล: `{model_status}`")
else:
    st.error(f"❌ สถานะโมเดล: `{model_status}`")

col_input, col_result = st.columns([1.2, 0.8], gap="large")

with col_input:
    st.subheader("📋 ระบุคุณลักษณะของบ้าน")
    
    # ---------------- ตั้งค่า Default ตรงนี้ให้ตรงกับแถว 0 ในชุดข้อมูล (ราคาประเมินควรออกมาราวๆ 208,500) ----------------
    with st.expander("1. พื้นที่หลักและขนาดที่ดิน (Main Area & Lot)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            gr_liv_area = st.number_input("พื้นที่ใช้สอยรวม (GrLivArea)", min_value=300, max_value=10000, value=1710, step=50)
            first_flr_sf = st.number_input("พื้นที่ชั้น 1 (1stFlrSF)", min_value=300, max_value=5000, value=856, step=50)
            second_flr_sf = st.number_input("พื้นที่ชั้น 2 (2ndFlrSF)", min_value=0, max_value=5000, value=854, step=50)
        with c2:
            lot_area = st.number_input("ขนาดที่ดิน (LotArea)", min_value=1000, max_value=50000, value=8450, step=100)
            lot_frontage = st.number_input("ความกว้างหน้าที่ดิน (LotFrontage)", min_value=0, max_value=400, value=65, step=5)
        with c3:
            lot_shape = st.selectbox("รูปทรงแปลงที่ดิน (Lot Shape)", options=list(shape_map.keys()), index=0)
            selected_subclass = st.selectbox("ประเภทของบ้าน (MSSubClass)", options=list(MSSUBCLASS_MAP.keys()))

    with st.expander("2. ห้องและสิ่งอำนวยความสะดวก (Rooms & Amenities)", expanded=True):
        c4, c5 = st.columns(2)
        with c4:
            bedroom = st.slider("ห้องนอน (BedroomAbvGr)", 0, 8, 3)
            tot_rms = st.slider("ห้องทั้งหมดไม่รวมห้องน้ำ", 2, 14, 8)
            kitchens = st.slider("จำนวนห้องครัว", 1, 3, 1)
        with c5:
            full_bath = st.slider("ห้องน้ำเต็มรูปแบบ (FullBath)", 0, 4, 2)
            half_bath = st.checkbox("มีห้องน้ำเล็ก (HalfBath)", value=True)
            has_fireplace = st.checkbox("มีเตาผิง (Fireplace)", value=False)
            FireplaceQu = st.selectbox("คุณภาพเตาผิง", list(qual_2_options.keys()), index=0, disabled=not has_fireplace) 

    with st.expander("3. ชั้นใต้ดิน (Basement)", expanded=True):
        c6, c7, c8 = st.columns(3)
        with c6:
            bsmt_fin_sf1 = st.number_input("พท.ใต้ดินส่วนตกแต่ง (BsmtFinSF1)", min_value=0, max_value=5000, value=706, step=50)
            bsmt_unf_sf = st.number_input("พท.ใต้ดินไม่ได้ตกแต่ง (BsmtUnfSF)", min_value=0, max_value=5000, value=150, step=50)
            has_bsmt = (bsmt_fin_sf1 + bsmt_unf_sf) > 0
        with c7:
            has_bsmt_full = st.checkbox("ห้องน้ำเต็มรูปแบบใต้ดิน", value=True, disabled=not has_bsmt)
            has_bsmt_half = st.checkbox("ห้องน้ำเล็กใต้ดิน", value=False, disabled=not has_bsmt)
        with c8:
            bsmt_qual = st.selectbox("คุณภาพโครงสร้างห้องใต้ดิน", list(qual_1_options.keys()), index=4, disabled=not has_bsmt) # Default: Good
            bsmt_exposure_th = st.selectbox("การเปิดรับแสงของห้องใต้ดิน", options=list(bsmt_exposure_options.keys()), index=1, disabled=not has_bsmt) # Default: No
            bsmt_fin_type_th = st.selectbox("ระดับการตกแต่งห้องใต้ดิน", options=list(bsmt_fintype_options.keys()), index=6, disabled=not has_bsmt) # Default: GLQ

    with st.expander("4. โรงจอดรถและภายนอก (Garage & Exterior)", expanded=True):
        c9, c10, c11 = st.columns(3)
        with c9:
            garage_cars = st.slider("ความจุจอดรถ (คัน)", 0, 5, 2)
            garage_area = st.number_input("พื้นที่โรงจอดรถ (GarageArea)", min_value=0, max_value=2000, value=548, step=50)
            garage_type_th = st.selectbox("ประเภทโรงจอดรถ", options=list(garage_type_map.keys()), index=0 if garage_cars > 0 else 5)
            garage_finish_th = st.selectbox("สภาพการตกแต่งโรงจอดรถ", options=list(garage_finish_options.keys()), index=2 if garage_cars > 0 else 0) # Default: RFn
        with c10:
            porch_area = st.number_input("พื้นที่ชานบ้าน/ระเบียงรวม", min_value=0, max_value=2000, value=0, step=20)
            has_open_porch = st.checkbox("มีระเบียงเปิดโล่ง", value=True)
            has_enclosed_porch = st.checkbox("มีระเบียงกระจก/ทึบ", value=False)
            has_screen_porch = st.checkbox("มีระเบียงมุ้งลวด", value=False)
        with c11:
            has_3ssn_porch = st.checkbox("มีระเบียง 3 ฤดู", value=False)
            has_wood_deck = st.checkbox("มีระเบียงไม้ (WoodDeck)", value=False)
            has_pool = st.checkbox("มีสระว่ายน้ำ (PoolArea)", value=False)

    with st.expander("5. วัสดุ ทำเล และคุณภาพ (Materials, Location & Quality)", expanded=True):
        c12, c13 = st.columns(2)
        with c12:
            selected_neighborhood = st.selectbox("ย่านที่ตั้งของบ้าน (Neighborhood)", options=list(NEIGHBORHOOD_MAP.keys()), index=list(NEIGHBORHOOD_MAP.keys()).index('CollgCr'))
            ms_zoning_th = st.selectbox(
                "โซนผังเมือง (MSZoning)", 
                ["ที่อยู่อาศัยหนาแน่นต่ำ (บ้านเดี่ยวทั่วไป)", "ที่อยู่อาศัยหนาแน่นปานกลาง (ทาวน์เฮาส์)", 
                 "โครงการหมู่บ้านจัดสรรริมน้ำ", "ที่อยู่อาศัยหนาแน่นสูง (คอนโด)"]
            )
            house_age = st.number_input("อายุของบ้าน (ปี)", min_value=0, max_value=150, value=5)
            is_remodeled = st.checkbox("เคยได้รับการรีโนเวท", value=True)
            remod_age = st.number_input("อายุหลังจากการรีโนเวท (ปี)", min_value=0, max_value=150, value=5, disabled=not is_remodeled)
            
            foundation_th = st.selectbox("ประเภทฐานราก (Foundation)", options=list(foundation_map.keys()), index=0)
            roof_style_th = st.selectbox("รูปทรงหลังคา (Roof Style)", options=list(roof_map.keys()), index=0)
        with c13:
            overall_qual = st.slider("คุณภาพรวม (OverallQual)", 1, 10, 7)
            overall_cond = st.slider("สภาพรวม (OverallCond)", 1, 10, 5)
            exter_qual = st.selectbox("คุณภาพวัสดุ", list(qual_1_options.keys()), index=0) # Good
            kitchen_qual = st.selectbox("คุณภาพห้องครัว", list(qual_1_options.keys()), index=4) # Good
            heating_qc = st.selectbox("คุณภาพระบบทำความร้อน", list(qual_2_options.keys()), index=3) # Good
            
            selected_ext1 = st.selectbox("วัสดุภายนอก 1", options=list(EXTERIOR1ST_MAP.keys()), index=list(EXTERIOR1ST_MAP.keys()).index('VinylSd'))
            selected_ext2 = st.selectbox("วัสดุภายนอก 2", options=list(EXTERIOR2ND_MAP.keys()), index=list(EXTERIOR2ND_MAP.keys()).index('VinylSd'))
            mas_vnr_th = st.selectbox("วัสดุตกแต่งผนังภายนอก", options=list(mas_vnr_map.keys()), index=list(mas_vnr_map.keys()).index("กรุอิฐโชว์แนว (BrkFace)"))
            
    btn_predict = st.button("🔮 คำนวณราคาประเมิน (Predict)", type="primary", use_container_width=True)

with col_result:
    st.subheader("📊 ผลการประเมินราคา")
    st.markdown("#### 🗺️ ที่ตั้งและทำเลของย่านที่เลือก")
    st.caption(f"แสดงตำแหน่งของย่าน: **{selected_neighborhood}**")
    
    map_obj = render_neighborhood_map(selected_neighborhood)
    st_folium(map_obj, width=450, height=300)

    if btn_predict:
        if model is None:
            st.error("โมเดลไม่พร้อมใช้งาน กรุณาตรวจสอบไฟล์ .pkl")
        else:
            # ใช้ฟีเจอร์จากตัวโมเดลเป็นหลัก
            if hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)
            else:
                expected_features = FEATURE_NAMES
                
            row_data = {col: 0.0 for col in expected_features}

            zoning_map = {
                "ที่อยู่อาศัยหนาแน่นต่ำ (บ้านเดี่ยวทั่วไป)": "RL",
                "ที่อยู่อาศัยหนาแน่นปานกลาง (ทาวน์เฮาส์)": "RM",
                "โครงการหมู่บ้านจัดสรรริมน้ำ": "FV",
                "ที่อยู่อาศัยหนาแน่นสูง (คอนโด)": "RH"
            }
            
            selected_shape_code = shape_map[lot_shape]
            selected_roof_code = roof_map[roof_style_th]
            selected_zoning = zoning_map[ms_zoning_th]
            selected_foundation_code = foundation_map[foundation_th]
            selected_garage_code = garage_type_map[garage_type_th]
            selected_mas_vnr_code = mas_vnr_map[mas_vnr_th]
            
            # คำนวณค่าต่างๆ อัตโนมัติ
            calc_total_bsmt_sf = float(bsmt_fin_sf1 + bsmt_unf_sf)
            calc_garage_yr_blt = 1.0 if garage_area > 0 else 0.0
            
            value_map = {
                'GrLivArea': float(gr_liv_area),
                'LotArea': float(lot_area),
                '1stFlrSF': float(first_flr_sf),
                '2ndFlrSF': float(second_flr_sf),
                'TotalBsmtSF': calc_total_bsmt_sf,
                'BsmtFinSF1': float(bsmt_fin_sf1),
                'BsmtUnfSF': float(bsmt_unf_sf),
                'GarageArea': float(garage_area),
                'GarageYrBlt': calc_garage_yr_blt,
                'BedroomAbvGr': float(bedroom),
                'TotRmsAbvGrd': float(tot_rms),
                'FullBath': float(full_bath),
                'GarageCars': float(garage_cars),
                'HouseAge': float(house_age),
                'RemodAge': float(remod_age) if is_remodeled else float(house_age),
                'IsRemodeled': 1.0 if is_remodeled else 0.0,
                
                # Binned & Binary Features
                'KitchenAbvGr_binned': float(kitchens),
                'BsmtHalfBath_binned': 1.0 if has_bsmt_half else 0.0,
                'BsmtFullBath_binned': 1.0 if has_bsmt_full else 0.0,
                'HalfBath_binned': 1.0 if half_bath else 0.0,
                'Fireplaces_binned': 1.0 if has_fireplace else 0.0,
                'OverallCond_binned': float(overall_cond),
                'OverallQual_binned': float(overall_qual),
                
                'TotalPorchSF': float(porch_area),
                'OpenPorchSF_binary': 1.0 if has_open_porch else 0.0,
                'ScreenPorch_binary': 1.0 if has_screen_porch else 0.0,
                'EnclosedPorch_binary': 1.0 if has_enclosed_porch else 0.0,
                '3SsnPorch_binary': 1.0 if has_3ssn_porch else 0.0,
                'WoodDeckSF_binary': 1.0 if has_wood_deck else 0.0,
                'PoolArea_binary': 1.0 if has_pool else 0.0,
                
                # ค่าที่ถูกซ่อนจาก UI ให้ค่าเป็น 0.0 ตามชุดข้อมูลจริงแถวที่ 0
                'BsmtFinSF2_binary': 0.0,
                'MiscVal_binary': 0.0,
                'LowQualFinSF_binary': 0.0,
                
                'MSSubClass': MSSUBCLASS_MAP[selected_subclass],
                'LotFrontage': float(lot_frontage),
                f'MSZoning_{selected_zoning}': 1.0,
                f'LotShape_{selected_shape_code}': 1.0,  
                'LotConfig_Inside': 1.0,
                f'RoofStyle_{selected_roof_code}': 1.0,
                f'Foundation_{selected_foundation_code}': 1.0,
                f'GarageType_{selected_garage_code}': 1.0,
                f'MasVnrType_{selected_mas_vnr_code}': 1.0,
                'MasVnrArea_binary': 0.0 if selected_mas_vnr_code == 'None' else 1.0,
                'Neighborhood': NEIGHBORHOOD_MAP[selected_neighborhood],
                'Exterior1st': EXTERIOR1ST_MAP[selected_ext1],
                'Exterior2nd': EXTERIOR2ND_MAP[selected_ext2],
                
                # --- ใช้ Mapping Ordinal ใหม่ที่มีค่า 0.0 (None) แล้ว ---
                'KitchenQual': qual_1_options[kitchen_qual],
                'ExterQual': qual_1_options[exter_qual],
                'HeatingQC': qual_2_options[heating_qc],
                'BsmtQual': qual_1_options[bsmt_qual] if calc_total_bsmt_sf > 0 else 0.0,
                'FireplaceQu': qual_2_options[FireplaceQu] if has_fireplace else -1.0,
                'GarageFinish': garage_finish_options[garage_finish_th] if garage_area > 0 else 0.0,
                'BsmtExposure': bsmt_exposure_options[bsmt_exposure_th] if calc_total_bsmt_sf > 0 else 0.0,
                'BsmtFinType1': bsmt_fintype_options[bsmt_fin_type_th] if calc_total_bsmt_sf > 0 else 0.0,
            } 
                
            for feature, val in value_map.items():
                if feature in row_data:
                    row_data[feature] = val

            target_style_col = SUBCLASS_TO_HOUSESTYLE.get(selected_subclass, None)
            if target_style_col and target_style_col in row_data:
                row_data[target_style_col] = 1.0

            # ดึงค่าตามลำดับ 87 Features 
            feature_values = [row_data.get(col, 0.0) for col in expected_features]
            input_array = np.array([feature_values], dtype=np.float64)

            safe_columns = [f"{col}_{i}" if list(expected_features).count(col) > 1 else col for i, col in enumerate(expected_features)]
            input_df = pd.DataFrame([feature_values], columns=safe_columns)

            try:
                raw_pred = float(model.predict(input_array)[0])
                if 0 < raw_pred < 30:
                    real_price = np.expm1(raw_pred)
                elif raw_pred <= 0:
                    real_price = 0.0
                else:
                    real_price = raw_pred

                st.session_state["predicted_price"] = real_price
                st.session_state["raw_pred"] = raw_pred
                st.session_state["input_df_display"] = input_df

            except Exception as e:
                st.error(f"Prediction Error: {e}")

    if st.session_state["predicted_price"] is not None:
        real_price = st.session_state["predicted_price"]
        raw_pred = st.session_state["raw_pred"]
        input_df = st.session_state["input_df_display"]

        st.metric(label="ราคาประเมินจริง (Estimated Sale Price)", value=f"${real_price:,.2f}")
        st.caption(f"ค่าดิบที่ได้จากโมเดล (Log Scale Output): `{raw_pred:.4f}`")

        if real_price <= 0:
            st.warning("⚠️ ผลลัพธ์ผิดปกติ: ลองปรับพื้นที่ใช้สอยให้มากขึ้น หรือลดอายุของบ้านลง")

        with st.expander("ดูตาราง Features (ที่ส่งเข้าโมเดล)"):
            st.dataframe(input_df.T, height=400)
    else:
        st.info("👈 ระบุรายละเอียดพื้นที่บ้านทางด้านซ้าย แล้วกดปุ่มเพื่อเริ่มประเมินราคา")
