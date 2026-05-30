import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['axes.unicode_minus'] = False

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample

import lightgbm as lgb
import shap

np.random.seed(42)


file_path = r"C:\Users\summe\Desktop\data\Data CO2.xlsx"#数据库读取
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()#删除列名空格
print("所有列名：", df.columns.tolist())

solvent_col = None#溶剂种类识别
for c in df.columns:
    if "solvent" in c.lower() and "type" in c.lower():
        solvent_col = c
        break

if solvent_col is None:
    raise ValueError("未找到 solvent type 列，请手动指定")
print(f"\n检测到溶剂类型列: '{solvent_col}'")

df = df[["SMILES", "T(k)", "P(Mpa)", "x1", solvent_col]].copy()#数据筛选
df.columns = ["SMILES", "T", "P", "x1", "Solvent_Type"]#统一列名
df["Solvent_Type"] = df["Solvent_Type"].astype(str).str.strip()#统一类别标签格式
df = df.dropna(subset=["SMILES", "T", "P", "x1"])#缺失值处理

print(f"\n清洗后数据集大小: {len(df)} 条")
print(f"\n溶剂类型分布:")
print(df["Solvent_Type"].value_counts())

#特征处理，SMILES转为分子结构
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * 10
    return [
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
    ]

rdkit_features = np.array([smiles_to_features(s) for s in df["SMILES"]])

#分子指纹处理
def smiles_to_ecfp(smiles, n_bits=256):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    return np.array(fp)

ecfp_features = np.array([smiles_to_ecfp(s) for s in df["SMILES"]])

macro_features = df[["T", "P"]].values
X = np.hstack([macro_features, rdkit_features, ecfp_features])
y = df["x1"].values

feature_names = (
    ["Temperature (T)", "Pressure (P)"] +
    ["MolWt", "LogP", "TPSA", "HBD", "HBA",
     "RotBonds", "ArRings", "HeavyAtoms", "FracCSP3", "RingCount"] +
    [f"ECFP_{i}" for i in range(256)]
)

print(f"\n特征矩阵维度: {X.shape}")


#数据集划分
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

solvent_types_all = df["Solvent_Type"].values
solvent_types_test = solvent_types_all[test_idx]

print(f"\n训练集: {len(X_train)} 条")
print(f"测试集: {len(X_test)} 条")
print(f"\n测试集溶剂类型分布:")
unique, counts = np.unique(solvent_types_test, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {u}: {c} 条")


#宏观+微观模型训练
print("\n" + "=" * 60)
print("[训练] Stage 1: LightGBM")
#第一阶段模型训练
model_stage1 = lgb.LGBMRegressor(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=48,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
model_stage1.fit(X_train, y_train)

y_pred_train_1 = model_stage1.predict(X_train)
y_pred_test_1 = model_stage1.predict(X_test)
print(f"Stage 1 Test R²={r2_score(y_test, y_pred_test_1):.4f}")

print("[训练] Stage 2: Random Forest (残差修正)")
residual_train = y_train - y_pred_train_1
#第二阶段学习
model_stage2 = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
model_stage2.fit(X_train, residual_train)

residual_pred_test = model_stage2.predict(X_test)
y_final_test = y_pred_test_1 + residual_pred_test

r2_final = r2_score(y_test, y_final_test)
rmse_final = np.sqrt(mean_squared_error(y_test, y_final_test))
mae_final = mean_absolute_error(y_test, y_final_test)

print(f"\n[最终性能] Two-Stage Model")
print(f"R²={r2_final:.4f}")
print(f"RMSE={rmse_final:.4f}")
print(f"MAE={mae_final:.4f}")
print("="*60)

#SHAP构建
print("\n[SHAP]构建TreeExplainer...")
explainer_stage1 = shap.TreeExplainer(model_stage1)#创建SHAP解释器
shap_values_stage1 = explainer_stage1.shap_values(X_test)#每个特征对预测的贡献

explainer_stage2 = shap.TreeExplainer(model_stage2)#对两阶段的贡献进行融合
shap_values_stage2 = explainer_stage2.shap_values(X_test)

shap_values_combined = shap_values_stage1 + shap_values_stage2
mean_abs_shap = np.abs(shap_values_combined).mean(axis=0)

print(f"SHAP值矩阵维度:{shap_values_combined.shape}")

#提取全局Top-10特征
TOP_K = 10
top_indices = np.argsort(mean_abs_shap)[-TOP_K:][::-1]#按SHAP重要性排序
top_feature_names = [feature_names[i] for i in top_indices]
top_shap_values = shap_values_combined[:, top_indices]
top_X_test = X_test[:, top_indices]

print("\n"+"="*70)
print(f"全局 Top-{TOP_K}特征")
print("="*70)
print(f"{'Rank':<6} {'Feature':<25} {'Mean |SHAP|':<15}")
print("-"*50)
for rank, (idx, fname) in enumerate(zip(top_indices, top_feature_names), 1):
    print(f"{rank:<6}{fname:<25}{mean_abs_shap[idx]:<15.6f}")

#绘图 全局 Top-10 SHAP Summary Plot
print("\n[绘图] 图: 全局 Top-10 SHAP 特征重要性分布")
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(
    top_shap_values,
    top_X_test,
    feature_names=top_feature_names,
    show=False,
    plot_size=(10, 6),
    max_display=TOP_K
)

plt.title(f"Figure: Global Top-{TOP_K} SHAP Feature Importance\n(Two-Stage Model)",
          fontsize=14, pad=15)
plt.xlabel("SHAP Value (impact on model output)", fontsize=12)
plt.tight_layout()
plt.savefig("fig_shap_top10_summary.png", dpi=300, bbox_inches="tight")
plt.show()
print("  → fig_shap_top10_summary.png 已保存")


#绘图 T和P的SHAP-Dependence Plot
print("\n[绘图] 图: 关键特征 SHAP Dependence 图")

temp_idx = feature_names.index("Temperature (T)")
pres_idx = feature_names.index("Pressure (P)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
shap.dependence_plot(
    temp_idx, shap_values_combined, X_test,
    feature_names=feature_names,
    interaction_index=pres_idx,
    ax=ax, show=False
)
ax.set_title("(a) Temperature (T)", fontsize=13)
ax.set_xlabel("Temperature (K)", fontsize=11)
ax.set_ylabel("SHAP Value", fontsize=11)

ax = axes[1]
shap.dependence_plot(
    pres_idx, shap_values_combined, X_test,
    feature_names=feature_names,
    interaction_index=temp_idx,
    ax=ax, show=False
)
ax.set_title("(b) Pressure (P)", fontsize=13)
ax.set_xlabel("Pressure (MPa)", fontsize=11)
ax.set_ylabel("SHAP Value", fontsize=11)

plt.suptitle("Figure: SHAP Dependence Plot for Key Features",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("fig_shap_dependence.png", dpi=300, bbox_inches="tight")
plt.show()
print("  → fig_shap_dependence.png 已保存")


#绘图 Top-10 全局柱状图
print("\n[绘图] 图: Top-10 特征重要性柱状图")

fig, ax = plt.subplots(figsize=(10, 6))

colors = plt.cm.viridis(np.linspace(0.3, 0.9, TOP_K))

bars = ax.barh(
    range(TOP_K),
    [mean_abs_shap[i] for i in top_indices],
    color=colors,
    edgecolor='gray',
    linewidth=0.8
)

for i, (bar, idx) in enumerate(zip(bars, top_indices)):
    width = bar.get_width()
    ax.text(
        width + 0.002, bar.get_y() + bar.get_height() / 2,
        f'{mean_abs_shap[idx]:.4f}',
        ha='left', va='center', fontsize=10
    )

ax.set_yticks(range(TOP_K))
ax.set_yticklabels(top_feature_names, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
ax.set_title(f"Figure: Top-{TOP_K} Feature Importance (Global)",
             fontsize=14, pad=15)
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig("fig_shap_top10_bar.png", dpi=300, bbox_inches="tight")
plt.show()
print("  → fig_shap_top10_bar.png 已保存")


#五类溶剂的 SHAP 特征重要性分析
print("\n" + "=" * 70)
print("五类溶剂的 SHAP 特征重要性分析")
print("=" * 70)


#指定五类溶剂
all_types_in_data = df["Solvent_Type"].unique()
print(f"\n数据中所有溶剂类型: {all_types_in_data}")
target_types_candidates = ["alcohol", "ester", "ketone", "ether", "aromatic"]# 定义目标五类

# 自动匹配
target_types = []
for candidate in target_types_candidates:
    matched = False
    for actual in all_types_in_data:
        if actual.lower() == candidate.lower():
            target_types.append(actual)
            matched = True
            break
    if not matched:
        print(f"警告: 未找到类别 '{candidate}'")

if len(target_types) < 5:
    print(f"\n警告: 只找到 {len(target_types)} 个目标类别: {target_types}")
    print("尝试使用样本量最多的五类作为替代...")
    type_counts_test = pd.Series(solvent_types_test).value_counts()
    target_types = type_counts_test.head(5).index.tolist()

print(f"\n最终选定的五类溶剂:")
for t in target_types:
    n_total = (solvent_types_all == t).sum()
    n_test = (solvent_types_test == t).sum()
    print(f"{t}: 总计{n_total}条, 测试集{n_test}条")

# 配色 五类固定颜色
type_colors = {
    target_types[0]: "#E74C3C",   # 红
    target_types[1]: "#3498DB",   # 蓝
    target_types[2]: "#2ECC71",   # 绿
    target_types[3]: "#F39C12",   # 橙
    target_types[4]: "#9B59B6",   # 紫
}


#计算各类别在全部特征上的 SHAP 重要性 + Bootstrap
print("\n[计算] 各类别 SHAP 重要性 (含 Bootstrap 置信区间)...")
n_bootstrap = 200
shap_by_type = {}

for stype in target_types:
    mask = solvent_types_test == stype
    n_s = mask.sum()
    if n_s == 0:
        print(f"跳过{stype}: 测试集中无样本")
        continue
    shap_subset = shap_values_combined[mask]
    boot_means = []
    for b in range(n_bootstrap):
        idx_b = resample(np.arange(n_s), replace=True, random_state=b)
        boot_means.append(np.abs(shap_subset[idx_b]).mean(axis=0))
    boot_means = np.array(boot_means)
    shap_by_type[stype] = {
        "mean_abs": np.abs(shap_subset).mean(axis=0),
        "std": boot_means.std(axis=0),
        "ci_lo": np.percentile(boot_means, 2.5, axis=0),
        "ci_hi": np.percentile(boot_means, 97.5, axis=0),
        "n": n_s,
    }

print("→Bootstrap 完成")
print(f"成功计算了{len(shap_by_type)}个类别的SHAP 值")


# 绘图 五类溶剂 Top-10 特征热图
print("\n[绘图] 图: 五类溶剂 Top-10 特征热图")

heatmap_labels = []
heatmap_data = []

for t in target_types:
    if t in shap_by_type:
        heatmap_labels.append(f"{t} (n={shap_by_type[t]['n']})")
        heatmap_data.append(shap_by_type[t]["mean_abs"][top_indices])

if len(heatmap_data) == 0:
    print("错误: 没有可用数据生成热图")
else:
    heatmap_data = np.array(heatmap_data)
    fig, ax = plt.subplots(figsize=(13, max(6, len(heatmap_labels) * 1.2)))

    sns.heatmap(
        heatmap_data,
        xticklabels=top_feature_names,
        yticklabels=heatmap_labels,
        cmap="YlOrRd",
        annot=True,
        fmt=".4f",
        annot_kws={"fontsize": 10},
        cbar_kws={"label": "Mean |SHAP Value|"},
        linewidths=0.6,
        ax=ax
    )

    ax.set_title(f"Figure: Top-{TOP_K} Feature Importance across Five Solvent Types",
                 fontsize=14, pad=15)
    ax.set_xlabel("Features", fontsize=12)
    ax.set_ylabel("Solvent Type", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.savefig("fig_solvent_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("  → fig_solvent_heatmap.png 已保存")


#绘图 五类溶剂各自 Top-10 柱状图（带误差条）
print("\n[绘图] 图: 五类溶剂各自 Top-10 特征（带 Bootstrap 误差条）")
valid_types_for_plot = [t for t in target_types if t in shap_by_type]
if len(valid_types_for_plot) == 0:
    print("错误: 没有可用类别生成柱状图")
else:
    # 5个子图：3列2行
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for idx, stype in enumerate(valid_types_for_plot):
        if idx >= 6:
            break
            
        ax = axes[idx]
        data = shap_by_type[stype]
        means = data["mean_abs"]
        stds = data["std"]
        n_s = data["n"]

        local_top = np.argsort(means)[-TOP_K:][::-1]

        ax.barh(
            range(TOP_K),
            means[local_top],
            xerr=stds[local_top],
            color=type_colors.get(stype, "#95A5A6"),
            edgecolor="gray",
            linewidth=0.6,
            capsize=3,
            alpha=0.85
        )

        #数值标注
        for j, li in enumerate(local_top):
            val = means[li]
            ax.text(val + stds[li] + 0.002, j,
                    f'{val:.4f}', va='center', fontsize=9)

        ax.set_yticks(range(TOP_K))
        ax.set_yticklabels([feature_names[i] for i in local_top], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP Value| ± Std", fontsize=11)
        ax.set_title(f"{stype} (n={n_s})", fontsize=13, fontweight='bold',
                     color=type_colors.get(stype, "#2C3E50"))
        ax.grid(axis='x', alpha=0.3, linestyle='--')

    #隐藏多余子图
    for idx in range(len(valid_types_for_plot), 6):
        axes[idx].set_visible(False)

    plt.suptitle(
        f"Figure: Class-Specific Top-{TOP_K} Feature Importance\n"
        f"(Two-Stage Residual Learning Model with Bootstrap CI)",
        fontsize=15, y=0.995
    )
    plt.tight_layout()
    plt.savefig("fig_top10.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("→fig_top10.png 已保存")

# 绘图 仅结构特征的类间对比（排除 T 和 P）
print("\n[绘图] 图Figure: 仅结构特征的类间对比（排除T和P）")

# 排除 Temperature (idx=0) 和 Pressure (idx=1)
struct_idx = [i for i in range(len(feature_names)) if i not in [0, 1]]
struct_shap_global = mean_abs_shap[struct_idx]

# 取结构特征中的 Top-10
top_struct_local = np.argsort(struct_shap_global)[-TOP_K:][::-1]
top_struct_global = [struct_idx[i] for i in top_struct_local]
top_struct_names = [feature_names[i] for i in top_struct_global]

if len(valid_types_for_plot) == 0:
    print("错误: 没有可用类别生成结构特征图")
else:
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for idx, stype in enumerate(valid_types_for_plot):
        if idx >= 6:
            break
            
        ax = axes[idx]
        data = shap_by_type[stype]
        means = data["mean_abs"]
        stds = data["std"]
        n_s = data["n"]

        vals = means[top_struct_global]
        errs = stds[top_struct_global]

        ax.barh(
            range(TOP_K),
            vals,
            xerr=errs,
            color=type_colors.get(stype, "#95A5A6"),
            edgecolor="gray",
            linewidth=0.6,
            capsize=3,
            alpha=0.85
        )

        for j in range(TOP_K):
            ax.text(vals[j] + errs[j] + 0.001, j,
                    f'{vals[j]:.4f}', va='center', fontsize=9)

        ax.set_yticks(range(TOP_K))
        ax.set_yticklabels(top_struct_names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP Value| ± Std", fontsize=11)
        ax.set_title(f"{stype} (n={n_s}) — Structural Features Only",
                     fontsize=12, fontweight='bold', 
                     color=type_colors.get(stype, "#2C3E50"))
        ax.grid(axis='x', alpha=0.3, linestyle='--')

    # 隐藏多余子图
    for idx in range(len(valid_types_for_plot), 6):
        axes[idx].set_visible(False)

    plt.suptitle(
        f"Figure: Structural Feature Importance across Solvent Types\n"
        f"(Excluding Temperature and Pressure)",
        fontsize=15, y=0.995
    )
    plt.tight_layout()
    plt.savefig("fig_structure_only_by_solvent.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("  → fig_structure_only_by_solvent.png 已保存")


#绘图 五类溶剂在Top-10结构特征上的分组柱状图
print("\n[绘图]图: 五类溶剂统一结构特征分组对比")

if len(valid_types_for_plot) < 2:
    print("警告: 可用类别不足 2 个，跳过分组柱状图")
else:
    fig, ax = plt.subplots(figsize=(16, 7))

    x_pos = np.arange(TOP_K)
    n_types = len(valid_types_for_plot)
    bar_width = 0.75 / n_types
    
    #让柱子居中
    offsets = np.linspace(-(n_types-1)/2, (n_types-1)/2, n_types)

    for idx, stype in enumerate(valid_types_for_plot):
        data = shap_by_type[stype]
        vals = data["mean_abs"][top_struct_global]
        errs = data["std"][top_struct_global]

        ax.bar(
            x_pos + offsets[idx] * bar_width,
            vals,
            bar_width,
            yerr=errs,
            label=f"{stype} (n={data['n']})",
            color=type_colors.get(stype, "#95A5A6"),
            edgecolor="gray",
            linewidth=0.5,
            capsize=2,
            alpha=0.85
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(top_struct_names, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title(
        "Figure: Structural Feature Importance Comparison\n"
        "across Five Solvent Types (with Bootstrap 95% CI)",
        fontsize=14, pad=15
    )
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9, ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig("fig_grouped_bar_structure.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("  → fig_grouped_bar_structure.png 已保存")

