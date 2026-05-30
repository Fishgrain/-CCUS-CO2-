import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb

file_path = r"C:\Users\summe\data\Data CO2.xlsx"#数据文件地址
df = pd.read_excel(file_path)
print("原始列名：", df.columns)#确认识别列名

df.columns = df.columns.str.strip()#去掉列名里存在的空格符
df = df[["SMILES", "T(k)", "P(Mpa)", "x1", "y1"]]
df.columns = ["SMILES", "T", "P", "x1", "y1"]#统一列名

df = df.dropna()#处理缺失值

#识别SMILES转为分子特征
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0]*10#防止报错
    
    return [
        Descriptors.MolWt(mol),#分子量
        Descriptors.MolLogP(mol),#疏水性
        Descriptors.TPSA(mol),#极性表面积
        Descriptors.NumHDonors(mol),#氢键供体数
        Descriptors.NumHAcceptors(mol),#氢键受体数
        Descriptors.NumRotatableBonds(mol),#可旋转键数
        Descriptors.NumAromaticRings(mol),#芳香环数
        Descriptors.HeavyAtomCount(mol),#重原子数
        Descriptors.FractionCSP3(mol),#sp3碳比例
        Descriptors.RingCount(mol)#环数量
    ]

rdkit_features = np.array([smiles_to_features(s) for s in df["SMILES"]])

#分子指纹（ECFP4）处理
def smiles_to_ecfp(smiles, n_bits=256):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    return np.array(fp)

ecfp_features = np.array([smiles_to_ecfp(s) for s in df["SMILES"]])

#构建特征
macro_features = df[["T", "P"]].values#提取温度压力
X = np.hstack([macro_features, rdkit_features, ecfp_features])#拼接特征（宏观+分子结构）
y = df["x1"].values#提取CO2溶解度

#训练集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#第一阶段：LightGBM
model_stage1 = lgb.LGBMRegressor(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model_stage1.fit(X_train, y_train)#训练模型

y_pred_train_1 = model_stage1.predict(X_train)#训练集预测
y_pred_test_1 = model_stage1.predict(X_test)#测试集预测

#计算残差
residual_train = y_train - y_pred_train_1
residual_test = y_test - y_pred_test_1

#第二阶段：Random Forest
model_stage2 = RandomForestRegressor(
    n_estimators=600,
    max_depth=12,
    min_samples_split=5,
    random_state=42
)

model_stage2.fit(X_train, residual_train)

residual_pred_train = model_stage2.predict(X_train)
residual_pred_test = model_stage2.predict(X_test)


#最终预测，融合
y_final_train = y_pred_train_1 + residual_pred_train
y_final_test = y_pred_test_1 + residual_pred_test


#评估指标
def evaluate(y_true, y_pred, name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"\n{name}")
    print(f"R2:{r2:.5f}")
    print(f"RMSE:{rmse:.5f}")
    print(f"MAE:{mae:.5f}")


#输出结果
evaluate(y_test, y_final_test, "Two-Stage Model")
