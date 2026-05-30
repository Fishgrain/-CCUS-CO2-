import os#文件路径管理
import io#处理上传文件的二进制流
import joblib#加载训练好的模型
import numpy as np
import pandas as pd
import streamlit as st#整个网站框架
import plotly.graph_objects as go#绘制交互式图表

from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors, Descriptors

from utils import predict_with_bundle#输入DataFrame→模型预测→返回结果

#页面构建
st.set_page_config(
    page_title="CCUS 分子结构与 CO2 溶解度预测平台",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_DIR = "models"#模型文件夹
DEFAULT_DATA_PATH = os.path.join("..", "data", "sample_data.csv")
#模型必须的输入列
REQUIRED_COLS = [
    "SMILES",
    "T(k)",
    "P(Mpa)",
    "ESP_max_pos",
    "ESP_max_neg",
    "r_peak",
    "g_peak",
    "solvent_type"
]


#定义网页样式
def inject_css():
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
        }

        .stApp {
            background: linear-gradient(180deg, #eef3fb 0%, #f7f9fc 100%);
        }

        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2rem;
            max-width: 1450px;
        }

        h1, h2, h3, h4 {
            color: #153b73;
        }
#顶部横幅
        .top-banner {
            background: linear-gradient(90deg, #4f73d9 0%, #5d81e8 100%);
            color: white;
            border-radius: 26px;
            padding: 24px 28px;
            box-shadow: 0 14px 35px rgba(58, 90, 190, 0.22);
            border: 1px solid rgba(255,255,255,0.22);
            margin-bottom: 14px;
        }

        .top-banner-title {
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 8px;
            color: white;
        }

        .top-banner-subtitle {
            font-size: 0.98rem;
            line-height: 1.8;
            color: rgba(255,255,255,0.95);
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border: 1.4px solid #d7e1f2;
            border-radius: 24px;
            padding: 18px 18px 16px 18px;
            box-shadow: 0 10px 28px rgba(32, 66, 120, 0.08);
        }
#模块标题
        .section-pill {
            display: inline-block;
            background: linear-gradient(90deg, #4f73d9 0%, #5f83ea 100%);
            color: white;
            font-weight: 700;
            font-size: 1.05rem;
            padding: 9px 16px;
            border-radius: 999px;
            margin-bottom: 14px;
            box-shadow: 0 6px 18px rgba(79, 115, 217, 0.22);
        }

        .soft-note {
            background: linear-gradient(180deg, #f6f9ff 0%, #eef4ff 100%);
            border: 1px solid #d9e4f7;
            border-radius: 16px;
            padding: 12px 14px;
            color: #355172;
            font-size: 0.93rem;
            line-height: 1.8;
            margin-top: 10px;
        }
#推荐结果框
        .recommend-box {
            background: linear-gradient(90deg, #edf8ef 0%, #eef6ff 100%);
            border: 1px solid #cfe3d2;
            border-left: 6px solid #2e7d32;
            border-radius: 16px;
            padding: 16px 18px;
            color: #183d6b;
            font-size: 1.02rem;
            font-weight: 600;
            line-height: 1.8;
            margin: 10px 0 16px 0;
        }

        .mini-caption {
            color: #546b8a;
            font-size: 0.9rem;
            line-height: 1.75;
        }

        .smiles-chip {
            display: inline-block;
            background: #eef4ff;
            color: #204576;
            border: 1px solid #d8e4f8;
            border-radius: 12px;
            padding: 8px 12px;
            font-size: 0.92rem;
            line-height: 1.6;
            word-break: break-all;
            margin-top: 8px;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid #dce6f7;
            border-radius: 16px;
            padding: 10px 12px;
            box-shadow: 0 6px 18px rgba(18, 58, 99, 0.05);
        }

        div[data-testid="stMetricLabel"] {
            color: #4e6480 !important;
            font-weight: 600 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #123a63 !important;
            font-weight: 800 !important;
        }

        .stButton>button {
            background: linear-gradient(90deg, #1f4e79 0%, #2e7d32 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.62rem 1.05rem;
            font-weight: 700;
            box-shadow: 0 8px 18px rgba(31, 78, 121, 0.18);
        }

        .stDownloadButton>button {
            background: linear-gradient(90deg, #2457a6 0%, #2e7d32 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.62rem 1.05rem;
            font-weight: 700;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 42px;
            border-radius: 12px 12px 0 0;
            padding-left: 14px;
            padding-right: 14px;
            background: #f3f7ff;
            color: #2d4f79;
            border: 1px solid #dde7f8;
        }

        .stTabs [aria-selected="true"] {
            background: #ffffff !important;
            color: #133b74 !important;
            font-weight: 700;
        }

        .stSelectbox label,
        .stMultiSelect label,
        .stSlider label,
        .stTextInput label {
            color: #24476f !important;
            font-weight: 700 !important;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid #dce6f5;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fbff 0%, #f3f8f4 100%);
            border-right: 1px solid rgba(18,58,99,0.08);
        }

        .footer-note {
            color: #5d7088;
            font-size: 0.9rem;
            line-height: 1.75;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


inject_css()


#缓存模型
@st.cache_resource
def load_bundle(bundle_path):
    return joblib.load(bundle_path)


def get_bundle_path(target_col):
    return os.path.join(MODEL_DIR, f"two_stage_{target_col}.pkl")


def bundle_exists(target_col):
    return os.path.exists(get_bundle_path(target_col))

#读取上传文件
def read_table_from_bytes(file_name, content):
    file_name = file_name.lower()

    if file_name.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(content), encoding="gbk")
    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(content))
    else:
        raise ValueError("只支持 csv / xlsx / xls 文件")

    df.columns = df.columns.astype(str).str.strip()
    return df


@st.cache_data
def read_table_from_path(path):
    if not os.path.exists(path):
        return None

    lower = path.lower()
    if lower.endswith(".csv"):
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="gbk")
    elif lower.endswith(".xlsx") or lower.endswith(".xls"):
        df = pd.read_excel(path)
    else:
        return None

    df.columns = df.columns.astype(str).str.strip()
    return df

#获取SMILES
def get_unique_smiles_from_data(df):
    if df is None or "SMILES" not in df.columns:
        return []

    smiles_list = (
        df["SMILES"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )
    smiles_list = [x for x in smiles_list if x.lower() not in ["nan", "none", "null"]]
    return smiles_list

# RDKit SMILES→Mol对象
def smiles_to_mol(smiles):
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None

#生成分子结构图
def mol_to_image(smiles, size=(520, 340)):
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None

    try:
        return Draw.MolToImage(mol, size=size, kekulize=True)
    except Exception:
        try:
            return Draw.MolToImage(mol, size=size, kekulize=False)
        except Exception:
            return None

#分子描述符，展示在右侧信息面板
def calc_mol_info(smiles):
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None

    try:
        return {
            "Formula": rdMolDescriptors.CalcMolFormula(mol),
            "MolWt": round(Descriptors.MolWt(mol), 4),
            "MolLogP": round(Descriptors.MolLogP(mol), 4),
            "TPSA": round(Descriptors.TPSA(mol), 4),
            "HBD": int(Descriptors.NumHDonors(mol)),
            "HBA": int(Descriptors.NumHAcceptors(mol)),
            "RotB": int(Descriptors.NumRotatableBonds(mol)),
            "AromaticRings": int(Descriptors.NumAromaticRings(mol)),
            "HeavyAtoms": int(Descriptors.HeavyAtomCount(mol)),
            "FractionCSP3": round(Descriptors.FractionCSP3(mol), 4),
        }
    except Exception:
        return None


def get_mol_descriptor_table(smiles):
    info = calc_mol_info(smiles)
    if info is None:
        return None
    return pd.DataFrame(list(info.items()), columns=["Descriptor", "Value"])


def smiles_label(smiles, max_len=36):
    s = str(smiles)
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


# 输入数据对齐，确保输入数据满足模型要求
def align_input_df(df, default_solvent_smiles="CCO"):
    out = df.copy()
    out.columns = out.columns.astype(str).str.strip()
#填充初始值，程序根据数据集自动补充ESP_max_pos/ESP_max_neg/r_peak/g_peak
    default_map = {
        "SMILES": default_solvent_smiles,
        "T(k)": 298.15,
        "P(Mpa)": 1.0,
        "ESP_max_pos": 0.0,
        "ESP_max_neg": 0.0,
        "r_peak": 0.0,
        "g_peak": 0.0,
        "solvent_type": default_solvent_smiles,
    }

    for col in REQUIRED_COLS:
        if col not in out.columns:
            out[col] = default_map[col]

    for col in ["T(k)", "P(Mpa)", "ESP_max_pos", "ESP_max_neg", "r_peak", "g_peak"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default_map[col])

    out["SMILES"] = (
        out["SMILES"]
        .astype(str)
        .replace("nan", default_solvent_smiles)
        .replace("None", default_solvent_smiles)
        .fillna(default_solvent_smiles)
    )

    out["solvent_type"] = (
        out["solvent_type"]
        .astype(str)
        .replace("nan", default_solvent_smiles)
        .replace("None", default_solvent_smiles)
        .fillna(default_solvent_smiles)
    )

    return out

#自动寻找预测结果列
def find_prediction_column(pred_df, target_col):
    candidates = [
        f"{target_col}_final_pred",
        "final_pred",
        "prediction",
        "pred",
        "Predict",
        "Prediction",
        target_col
    ]

    for c in candidates:
        if c in pred_df.columns:
            return c

    suffix_cols = [c for c in pred_df.columns if str(c).endswith("_final_pred")]
    if suffix_cols:
        return suffix_cols[0]

    lgb_candidates = [f"{target_col}_lgb_pred", "lgb_pred", "lgbm_pred"]
    residual_candidates = [f"{target_col}_residual_pred", "residual_pred", "rf_residual_pred"]

    lgb_col = next((c for c in lgb_candidates if c in pred_df.columns), None)
    res_col = next((c for c in residual_candidates if c in pred_df.columns), None)

    if lgb_col is not None and res_col is not None:
        final_col = f"{target_col}_final_pred"
        pred_df[final_col] = pred_df[lgb_col] + pred_df[res_col]
        return final_col

    exclude_cols = set(REQUIRED_COLS)
    numeric_cols = [
        c for c in pred_df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(pred_df[c])
    ]

    if numeric_cols:
        return numeric_cols[-1]

    raise ValueError("无法在预测结果中找到预测值列，请检查 utils.py 中 predict_with_bundle() 的输出格式。")

#预测主函数，输入数据→对齐→调用模型→寻找预测列→返回结果
def predict_df(bundle, df, target_col, default_solvent_smiles="CCO"):
    df_ready = align_input_df(df, default_solvent_smiles=default_solvent_smiles)
    pred_raw = predict_with_bundle(df_ready, bundle)

    if isinstance(pred_raw, pd.Series):
        pred_df = pd.DataFrame({"prediction": pred_raw.values})
    elif isinstance(pred_raw, np.ndarray):
        arr = np.asarray(pred_raw)
        if arr.ndim == 1:
            pred_df = pd.DataFrame({"prediction": arr})
        else:
            pred_df = pd.DataFrame(arr)
    elif isinstance(pred_raw, pd.DataFrame):
        pred_df = pred_raw.copy()
    else:
        pred_df = pd.DataFrame({"prediction": pred_raw})

    for col in df_ready.columns:
        if col not in pred_df.columns:
            pred_df[col] = df_ready[col].values

    pred_col = find_prediction_column(pred_df, target_col)
    standard_col = f"{target_col}_final_pred"

    if standard_col not in pred_df.columns:
        pred_df[standard_col] = pd.to_numeric(pred_df[pred_col], errors="coerce")

    return df_ready, pred_df, standard_col

#生成一行预测数据
def make_sample_row(solvent_smiles, t_val, p_val):
    return {
        "SMILES": str(solvent_smiles),
        "T(k)": float(t_val),
        "P(Mpa)": float(p_val),
        "ESP_max_pos": 0.0,
        "ESP_max_neg": 0.0,
        "r_peak": 0.0,
        "g_peak": 0.0,
        "solvent_type": str(solvent_smiles),
    }


# 绘图
def apply_plot_style(fig, title=None):
    fig.update_layout(
        template="plotly_white",
        title=title,
        font=dict(
            family="Segoe UI, Microsoft YaHei, sans-serif",
            size=13,
            color="#14355c"
        ),
        margin=dict(l=20, r=20, t=55, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(31,78,121,0.10)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(31,78,121,0.10)", zeroline=False)
    return fig

#等高线绘图，固定SMILES→生成T-P网络→批量预测→构建等高线→寻找最大值
def build_tp_contour(
    bundle,
    target_col,
    solvent_smiles,
    t_center,
    p_center,
    t_range,
    p_range,
    n_t=35,
    n_p=35
):
    t_min, t_max = t_range
    p_min, p_max = p_range

    t_values = np.linspace(t_min, t_max, n_t)
    p_values = np.linspace(p_min, p_max, n_p)

    rows = []
    for p in p_values:
        for t in t_values:
            rows.append(make_sample_row(solvent_smiles, t, p))

    grid_df = pd.DataFrame(rows)
    _, pred_df, pred_col = predict_df(
        bundle=bundle,
        df=grid_df,
        target_col=target_col,
        default_solvent_smiles=solvent_smiles
    )

    pred_df[pred_col] = pd.to_numeric(pred_df[pred_col], errors="coerce")

    if pred_df[pred_col].isna().all():
        raise ValueError(f"{solvent_smiles} 的预测结果全部为空，无法绘图。")

    z = pred_df[pred_col].values.reshape(len(p_values), len(t_values))

    current_df = pd.DataFrame([make_sample_row(solvent_smiles, t_center, p_center)])
    _, current_pred_df, current_pred_col = predict_df(
        bundle=bundle,
        df=current_df,
        target_col=target_col,
        default_solvent_smiles=solvent_smiles
    )

    current_pred = float(current_pred_df[current_pred_col].iloc[0])

    valid_pred_df = pred_df.dropna(subset=[pred_col]).copy()
    best_idx = valid_pred_df[pred_col].idxmax()
    best_row = valid_pred_df.loc[best_idx]

    best_t = float(best_row["T(k)"])
    best_p = float(best_row["P(Mpa)"])
    best_pred = float(best_row[pred_col])

    fig = go.Figure()
#等高线
    fig.add_trace(
        go.Contour(
            x=t_values,
            y=p_values,
            z=z,
            colorscale="Viridis",
            contours=dict(showlines=True, coloring="heatmap"),
            line_smoothing=0.85,
            colorbar=dict(title="预测值"),
            hovertemplate=(
                "T=%{x:.2f} K<br>"
                "P=%{y:.2f} Mpa<br>"
                "预测值=%{z:.6f}<extra></extra>"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[t_center],
            y=[p_center],
            mode="markers+text",
            marker=dict(
                size=13,
                color="#ff5b4d",
                symbol="circle",
                line=dict(color="white", width=1)
            ),
            text=[f"当前点<br>{current_pred:.4f}"],
            textposition="top center",
            textfont=dict(color="#111", size=12),
            hovertemplate=(
                f"当前条件<br>T={t_center:.2f} K<br>P={p_center:.2f} Mpa<br>"
                f"预测值={current_pred:.6f}<extra></extra>"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[best_t],
            y=[best_p],
            mode="markers+text",
            marker=dict(
                size=15,
                color="#00c853",
                symbol="star",
                line=dict(color="white", width=1)
            ),
            text=[f"最优点<br>{best_pred:.4f}"],
            textposition="bottom center",
            textfont=dict(color="#111", size=12),
            hovertemplate=(
                f"最优条件<br>T={best_t:.2f} K<br>P={best_p:.2f} Mpa<br>"
                f"预测值={best_pred:.6f}<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title=f"溶剂 SMILES: {smiles_label(solvent_smiles, 45)}",
        xaxis_title="T (K)",
        yaxis_title="P (Mpa)",
        height=500,
        showlegend=False
    )

    return apply_plot_style(fig), current_pred, best_t, best_p, best_pred


def safe_download_csv(df):
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

#自动推荐CO2在该溶剂的x温度、y压力时溶解度达到最佳状态
def build_recommendation(summary_df):
    best_idx = summary_df["最优预测值"].idxmax()
    row = summary_df.loc[best_idx]
    return (
        f"推荐你在 **{row['最佳温度(K)']:.2f} K** 温度、"
        f"**{row['最佳压力(Mpa)']:.2f} Mpa** 压力下选择 "
        f"**{row['溶剂SMILES']}** 溶剂；"
        f"该条件下模型预测值为 **{row['最优预测值']:.6f}**。"
    )


# 顶部横幅
st.markdown(
    """
    <div class="top-banner">
        <div class="top-banner-title">CCUS 分子结构与 CO₂ 溶解度预测平台</div>
        <div class="top-banner-subtitle">
            基于数据文件中 SMILES 列的候选溶剂进行筛选，分别生成分子结构图与 T-P 等高线预测图，
            并自动给出最优溶剂与推荐条件。
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# 侧边栏制作，预测目标为x1、y1、可以上传数据集，给温度和压力做一个搜索范围，规定T-P网格密度
st.sidebar.markdown("## 模型设置")
target_col = st.sidebar.selectbox("选择预测目标", ["x1", "y1"], index=0)

bundle_path = get_bundle_path(target_col)
if bundle_exists(target_col):
    bundle = load_bundle(bundle_path)
    st.sidebar.success(f"已加载模型：{bundle_path}")
else:
    bundle = None
    st.sidebar.error(f"未找到模型文件：{bundle_path}")

st.sidebar.markdown("---")
st.sidebar.markdown("## 数据文件设置")

data_path = st.sidebar.text_input("默认数据文件路径", value=DEFAULT_DATA_PATH)
uploaded_data_file = st.sidebar.file_uploader("或者上传数据文件", type=["csv", "xlsx", "xls"])

data_df = None
if uploaded_data_file is not None:
    try:
        data_df = read_table_from_bytes(uploaded_data_file.name, uploaded_data_file.getvalue())
        st.sidebar.success(f"已读取上传文件：{uploaded_data_file.name}")
    except Exception as e:
        st.sidebar.error(f"上传文件读取失败：{e}")
else:
    data_df = read_table_from_path(data_path)
    if data_df is not None:
        st.sidebar.success(f"已读取默认数据：{data_path}")
    else:
        st.sidebar.warning("未读取到默认数据文件，请检查路径或上传文件。")

solvent_smiles_options = get_unique_smiles_from_data(data_df)
if len(solvent_smiles_options) == 0:
    solvent_smiles_options = ["CCO"]
    st.sidebar.warning("数据文件中未找到有效的 SMILES 列，已使用默认 CCO。")
else:
    st.sidebar.info(f"从数据中读取到 {len(solvent_smiles_options)} 个候选 SMILES。")

st.sidebar.markdown("---")
st.sidebar.markdown("## 等高线搜索范围")
t_window = st.sidebar.slider("温度搜索半宽 ΔT(K)", 5.0, 100.0, 30.0, 5.0)
p_window = st.sidebar.slider("压力搜索半宽 ΔP(Mpa)", 1.0, 15.0, 5.0, 1.0)
grid_n = st.sidebar.slider("等高线网格密度", 15, 60, 35, 5)


# 保存用户状态
if "t_val" not in st.session_state:
    st.session_state.t_val = 298.15

if "p_val" not in st.session_state:
    st.session_state.p_val = 1.0

if "selected_solvent_smiles" not in st.session_state:
    st.session_state.selected_solvent_smiles = solvent_smiles_options[:3]

st.session_state.selected_solvent_smiles = [
    s for s in st.session_state.selected_solvent_smiles
    if s in solvent_smiles_options
]

if len(st.session_state.selected_solvent_smiles) == 0:
    st.session_state.selected_solvent_smiles = solvent_smiles_options[:3]


# 上半部分
left_col, right_col = st.columns([1.0, 1.25], gap="large")

with left_col:
    with st.container(border=True):
        st.markdown('<div class="section-pill">参数输入</div>', unsafe_allow_html=True)

        with st.form("input_form"):
            t_val = st.slider(
                "T(k)",
                min_value=150.0,
                max_value=500.0,
                value=float(st.session_state.t_val),
                step=0.1
            )

            p_val = st.slider(
                "P(Mpa)",
                min_value=0.0,
                max_value=30.0,
                value=float(st.session_state.p_val),
                step=0.1
            )

            selected_solvent_smiles = st.multiselect(
                "选择溶剂 SMILES，可多选",
                options=solvent_smiles_options,
                default=st.session_state.selected_solvent_smiles,
                format_func=lambda x: smiles_label(x, 55)
            )

            submitted = st.form_submit_button("开始预测")

        st.markdown(
            """
            <div class="soft-note">
            溶剂列表来自数据文件中的 <b>SMILES</b> 列。<br>
            你可以一次选择多个候选溶剂，系统会分别输出每个溶剂的分子结构图和 T-P 等高线图，
            并在结果区给出综合推荐。
            </div>
            """,
            unsafe_allow_html=True
        )

        if submitted:
            st.session_state.t_val = float(t_val)
            st.session_state.p_val = float(p_val)
            st.session_state.selected_solvent_smiles = selected_solvent_smiles
            st.rerun()

with right_col:
    with st.container(border=True):
        st.markdown('<div class="section-pill">分子结构</div>', unsafe_allow_html=True)

        preview_smiles = (
            st.session_state.selected_solvent_smiles[0]
            if len(st.session_state.selected_solvent_smiles) > 0
            else solvent_smiles_options[0]
        )

        mol_col1, mol_col2 = st.columns([1.15, 0.85], gap="medium")

        with mol_col1:
            img = mol_to_image(preview_smiles, size=(520, 340))
            if img is not None:
                st.image(img, use_container_width=True)
            else:
                st.warning("当前选中的 SMILES 无法被 RDKit 解析，因此不能显示分子结构图。")

            st.markdown(
                f'<div class="smiles-chip"><b>当前预览 SMILES：</b> {preview_smiles}</div>',
                unsafe_allow_html=True
            )

        with mol_col2:
            desc_df = get_mol_descriptor_table(preview_smiles)
            if desc_df is not None:
                st.dataframe(desc_df, use_container_width=True, hide_index=True)
            else:
                st.info("暂无分子描述符可显示。")

        st.markdown(
            """
            <div class="soft-note">
            当前展示的是所选溶剂列表中的第一个 SMILES。<br>
            下方“预测结果”区域会分别展示每个溶剂的结构与预测图，因此这里主要用于快速预览。
            </div>
            """,
            unsafe_allow_html=True
        )


#预测结果
with st.container(border=True):
    st.markdown('<div class="section-pill">预测结果</div>', unsafe_allow_html=True)

    if bundle is None:
        st.error("当前未加载模型，请确认 models 文件夹下存在 two_stage_x1.pkl 或 two_stage_y1.pkl。")

    elif len(st.session_state.selected_solvent_smiles) == 0:
        st.warning("请至少选择一个溶剂 SMILES。")

    else:
        t_center = float(st.session_state.t_val)
        p_center = float(st.session_state.p_val)

        t_min = max(150.0, t_center - float(t_window))
        t_max = min(500.0, t_center + float(t_window))
        p_min = max(0.0, p_center - float(p_window))
        p_max = min(30.0, p_center + float(p_window))

        summary_rows = []

        tab_titles = [smiles_label(s, 24) for s in st.session_state.selected_solvent_smiles]
        tabs = st.tabs(tab_titles)

        for solvent_smiles, tab in zip(st.session_state.selected_solvent_smiles, tabs):
            with tab:
                st.markdown(f"### 溶剂 SMILES：`{solvent_smiles}`")

                img_col, plot_col = st.columns([0.82, 1.68], gap="large")

                with img_col:
                    solvent_img = mol_to_image(solvent_smiles, size=(420, 320))
                    if solvent_img is not None:
                        st.image(solvent_img, use_container_width=True)
                    else:
                        st.warning("该 SMILES 无法解析，不能显示分子结构。")

                    desc_df = get_mol_descriptor_table(solvent_smiles)
                    if desc_df is not None:
                        st.dataframe(desc_df, use_container_width=True, hide_index=True)

                with plot_col:
                    try:
                        fig, current_pred, best_t, best_p, best_pred = build_tp_contour(
                            bundle=bundle,
                            target_col=target_col,
                            solvent_smiles=solvent_smiles,
                            t_center=t_center,
                            p_center=p_center,
                            t_range=(t_min, t_max),
                            p_range=(p_min, p_max),
                            n_t=int(grid_n),
                            n_p=int(grid_n)
                        )

                        m1, m2, m3 = st.columns(3)
                        m1.metric("当前条件预测值", f"{current_pred:.6f}")
                        m2.metric("最佳温度", f"{best_t:.2f} K")
                        m3.metric("最佳压力", f"{best_p:.2f} Mpa")

                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown(
                            f"""
                            <div class="soft-note">
                            在当前输入条件下，所选溶剂的预测值为 <b>{current_pred:.6f}</b>。<br>
                            在本次搜索范围内，推荐条件为 <b>{best_t:.2f} K</b>、<b>{best_p:.2f} Mpa</b>，
                            对应最优预测值为 <b>{best_pred:.6f}</b>。
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        summary_rows.append({
                            "溶剂SMILES": solvent_smiles,
                            "当前温度(K)": t_center,
                            "当前压力(Mpa)": p_center,
                            "当前条件预测值": current_pred,
                            "最佳温度(K)": best_t,
                            "最佳压力(Mpa)": best_p,
                            "最优预测值": best_pred
                        })

                    except Exception as e:
                        st.error(f"该溶剂预测或绘图失败：{e}")

        if len(summary_rows) > 0:
            summary_df = pd.DataFrame(summary_rows)

            st.markdown("### 溶剂推荐结果")
            rec_text = build_recommendation(summary_df)
            st.markdown(
                f'<div class="recommend-box">{rec_text}</div>',
                unsafe_allow_html=True
            )

            st.markdown("### 各溶剂预测对比表")
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.download_button(
                label="下载预测结果 CSV",
                data=safe_download_csv(summary_df),
                file_name=f"{target_col}_solvent_recommendation.csv",
                mime="text/csv"
            )
        else:
            st.warning("所有溶剂均预测失败，请检查模型输入列、SMILES 格式或 utils.py 输出格式。")


# 页脚
st.markdown("---")
st.markdown(
    """
    <div class="footer-note">
    本页面当前聚焦于：候选溶剂筛选、分子结构展示、T-P 等高线预测与推荐结果输出。<br>
    不再显示模型性能图，从而让页面结构更简洁、重点更聚焦。
    </div>
    """,
    unsafe_allow_html=True
)
