import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb


NUMERIC_INPUT_COLS = ["T(k)", "P(Mpa)", "ESP_max_pos", "ESP_max_neg", "r_peak", "g_peak"]
CAT_INPUT_COLS = ["solvent_type"]
SMILES_COL = "SMILES"

RDKIT_DESC_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "HeavyAtomCount",
    "FractionCSP3", "RingCount"
]

# 将SMILES字符串转换为 RDKit 分子描述符特征向量
def smiles_to_rdkit_features(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(len(RDKIT_DESC_NAMES), dtype=float)

    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.RingCount(mol)
    ], dtype=float)

#将SMILES字符串转换为Morgan ECFP指纹向量
def smiles_to_ecfp(smiles, n_bits=256):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(n_bits, dtype=float)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(float)

# 构建OneHotEncoder，兼容 sklearn 新旧版本 API
def _build_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)#遇到训练时未见的类别时忽略并返回密集矩阵
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)# 旧版 sklearn 没有sparse_output参数，捕获typeerror，回退到旧版参数名

#清洗数据框，确保必要列存在、类型转换、缺失值处理
def clean_dataframe(df, target_col="x1"):
    df = df.copy()

    # 确保必要列存在
    need_cols = [SMILES_COL] + NUMERIC_INPUT_COLS + CAT_INPUT_COLS + [target_col]
    for c in need_cols:
        if c not in df.columns:
            df[c] = np.nan

    # 数值列转为数值
    for c in NUMERIC_INPUT_COLS + [target_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")#无法转换的值转为NaN

    # 类别列缺失处理
    df["solvent_type"] = df["solvent_type"].fillna("Unknown").astype(str)

    # 删除目标缺失
    df = df.dropna(subset=[SMILES_COL, target_col]).reset_index(drop=True)

    return df

# 从原始数据中提取数值、分子描述符、指纹、类别特征；返回特征矩阵、特征名列表、编码器和填充器
def extract_features(df, encoder=None, imputer=None, fit=False):
    """
    返回:
    X, feature_names, encoder, imputer
    """
    df = df.copy()

    #数值特征
    X_num = df[NUMERIC_INPUT_COLS].apply(pd.to_numeric, errors="coerce").values

    if fit:
        imputer = SimpleImputer(strategy="median")
        X_num = imputer.fit_transform(X_num)
    else:
        X_num = imputer.transform(X_num)

    #分子描述符
    desc_list = [smiles_to_rdkit_features(smi) for smi in df[SMILES_COL]]
    X_desc = np.vstack(desc_list)

    #Morgan 指纹
    fp_list = [smiles_to_ecfp(smi, n_bits=256) for smi in df[SMILES_COL]]
    X_fp = np.vstack(fp_list)

    #类别特征
    X_cat = df[CAT_INPUT_COLS].fillna("Unknown").astype(str).values
    if fit:
        encoder = _build_encoder()
        X_cat_ohe = encoder.fit_transform(X_cat)
    else:
        X_cat_ohe = encoder.transform(X_cat)

    #拼接融合
    X = np.hstack([X_num, X_desc, X_fp, X_cat_ohe])

    #特征名
    feature_names = []
    feature_names.extend(NUMERIC_INPUT_COLS)
    feature_names.extend([f"RDKit_{n}" for n in RDKIT_DESC_NAMES])
    feature_names.extend([f"FP_{i}" for i in range(256)])

    try:
        cat_names = encoder.get_feature_names_out(CAT_INPUT_COLS).tolist()
    except Exception:
        cat_names = [f"solvent_type_{c}" for c in encoder.categories_[0].tolist()]
    feature_names.extend(cat_names)

    return X, feature_names, encoder, imputer

#计算回归模型的评估指标
def evaluate(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred)
    }

#训练两阶段模型，返回bundle和预测结果对比表
def train_two_stage_model(df, target_col="x1", random_state=42):
    df = clean_dataframe(df, target_col=target_col)

    X, feature_names, encoder, imputer = extract_features(df, fit=True)
    y = df[target_col].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=random_state
    )

    evals_result = {}

    lgb_model = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state
    )

    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        eval_names=["train", "valid"],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.record_evaluation(evals_result)
        ]
    )

    # 残差计算
    residual_train = y_train - lgb_model.predict(X_train)

    rf_model = RandomForestRegressor(
        n_estimators=600,
        max_depth=12,
        min_samples_split=5,
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X_train, residual_train)

    # 最终预测，模型融合
    y_train_pred = lgb_model.predict(X_train) + rf_model.predict(X_train)
    y_test_pred = lgb_model.predict(X_test) + rf_model.predict(X_test)

    metrics = {
        "train": evaluate(y_train, y_train_pred),
        "test": evaluate(y_test, y_test_pred)
    }

    result_df = pd.concat([
        pd.DataFrame({"set": "Train", "exp": y_train, "pred": y_train_pred}),
        pd.DataFrame({"set": "Test", "exp": y_test, "pred": y_test_pred}),
    ], ignore_index=True)

    bundle = {
        "target_col": target_col,
        "lgb_model": lgb_model,
        "rf_model": rf_model,
        "encoder": encoder,
        "imputer": imputer,
        "feature_names": feature_names,
        "evals_result": evals_result,
        "metrics": metrics
    }

    return bundle, result_df

# 使用已训练的模型包对新数据进行预测
def predict_with_bundle(df, bundle):
    df = df.copy()
    target_col = bundle["target_col"]

    # 预测时不要求有目标列
    for c in NUMERIC_INPUT_COLS + CAT_INPUT_COLS:
        if c not in df.columns:
            df[c] = np.nan
    if SMILES_COL not in df.columns:
        raise ValueError("输入数据缺少 SMILES 列")

    df["solvent_type"] = df["solvent_type"].fillna("Unknown").astype(str)

    X, _, _, _ = extract_features(
        df,
        encoder=bundle["encoder"],#复用训练时的编码器
        imputer=bundle["imputer"],#复用训练时的填充器
        fit=False
    )

    lgb_pred = bundle["lgb_model"].predict(X)
    rf_pred = bundle["rf_model"].predict(X)
    final_pred = lgb_pred + rf_pred

    out = df.copy()
    out[f"{target_col}_pred"] = final_pred
    return out
