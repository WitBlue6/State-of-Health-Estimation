'''
舵机1参数1	舵机1参数2	舵机1参数3	舵机1参数4	舵机1参数5	舵机1参数6	舵机2参数1	舵机2参数2	舵机2参数3	舵机2参数4	舵机2参数5	舵机2参数6	舵机3参数1	舵机3参数2	舵机3参数3	舵机3参数4	舵机3参数5	舵机3参数6	舵机4参数1	舵机4参数2	舵机4参数3	舵机4参数4	舵机4参数5	舵机4参数6	惯阻X轴加速度	惯阻X轴角速度	惯阻X轴姿态角	惯阻Y轴加速度	惯阻Y轴角速度	惯阻Y轴姿态角	惯阻Z轴加速度	惯阻Z轴角速度	惯阻Z轴姿态角	电源电压	电源电流	电源功率	电源电量	北斗经度	北斗纬度	北斗高度
6.974	0	0	1756	-17.5	0	7.073	0	0	1932	0.9	0	7.045	0	0	1947	45.8	0	6.987	0.016	0.106	1833	-16.9	0	-0.000236132	-0.004483876	0.002480302	-0.394561887	0.051412608	-9.761220932	-0.005255251	-0.04112225	3.819949389	7.01975	0.016	0.106	1867	3.261122605	0.078836204	141.0647754
6.828	0	0	1756	-4.8	0	6.947	0	0	1932	0.9	0	6.919	0	0	1947	45.8	0	6.929	0	0	1832	-3.7	0	0.000296501	-0.003951244	0.003545566	-0.441191912	0.011956421	-9.756439209	-0.00522137	-0.041122518	3.819844961	6.90575	0	0	1866.75	2.408084621	1.746787196	142.0950171
6.816	0	0	1756	1.4	0	6.922	0	0	1932	0.9	0	6.896	0	0	1947	45.8	0	6.891	0.048	0.321	1832	2.5	0	0.000829132	-0.00288598	0.00194767	-0.380214155	0.06934724	-9.798286438	-0.005217033	-0.041107211	3.819865465	6.88125	0.048	0.321	1866.75	2.494371244	1.736247936	143.2133688
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
import os
# 设置中文显示
sns.set_theme(style="whitegrid", font="Songti SC")
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['font.sans-serif'] = ['Songti SC']  
plt.rcParams['axes.unicode_minus'] = False  

# ===============================
# 1. 读取数据
# ===============================
# 假设是CSV文件，如果是Excel就换成 pd.read_excel("your_file.xlsx")
columns = [
    "舵机1参数1","舵机1参数2","舵机1参数3","舵机1参数4","舵机1参数5","舵机1参数6",
    "舵机2参数1","舵机2参数2","舵机2参数3","舵机2参数4","舵机2参数5","舵机2参数6",
    "舵机3参数1","舵机3参数2","舵机3参数3","舵机3参数4","舵机3参数5","舵机3参数6",
    "舵机4参数1","舵机4参数2","舵机4参数3","舵机4参数4","舵机4参数5","舵机4参数6",
    "惯阻X轴加速度","惯阻X轴角速度","惯阻X轴姿态角",
    "惯阻Y轴加速度","惯阻Y轴角速度","惯阻Y轴姿态角",
    "惯阻Z轴加速度","惯阻Z轴角速度","惯阻Z轴姿态角",
    "电源电压","电源电流","电源功率","电源电量",
    "北斗经度","北斗纬度","北斗高度"
]
df = pd.read_csv("./dataset/无异常.txt", header=None, names=columns)
X = df.select_dtypes(include="number")
# 创建输出目录
out_dir = "./data_quality_report"
os.makedirs(out_dir, exist_ok=True)

# ===============================
# 1. PCA 主成分解释度
# ===============================
pca = PCA()
pca.fit(X)
explained = pca.explained_variance_ratio_
cum_explained = np.cumsum(explained)

plt.figure(figsize=(8,6))
sns.barplot(x=list(range(1, len(explained)+1)), y=explained, palette="Blues_d")
plt.plot(range(1,len(explained)+1), cum_explained, marker='o', color='crimson', linewidth=2, label="累计解释度")
plt.xlabel("主成分序号")
plt.ylabel("解释方差比例")
plt.title("PCA 主成分解释度", weight="bold")
plt.legend()
plt.tight_layout()
plt.savefig(f"{out_dir}/pca_variance.png", dpi=300, bbox_inches="tight")
plt.close()

# ===============================
# 2. 信噪比 (SNR)
# ===============================
snr = 20 * np.log10(df.mean() / df.std(ddof=0))
snr = snr.replace([np.inf, -np.inf], np.nan)  # 去掉Inf
snr_valid = snr.dropna()  # 只用有效列

plt.figure(figsize=(12,6))
sns.barplot(x=snr_valid.sort_values().index, y=snr_valid.sort_values().values, palette="viridis")
plt.xticks(rotation=90)
plt.ylabel("SNR (均值/标准差)")
plt.title("各变量信噪比", weight="bold")
plt.tight_layout()
plt.savefig(f"{out_dir}/snr.png", dpi=300)
plt.close()

# ===============================
# 3. 变异系数 (CV)
# ===============================
cv = df.std(ddof=0) / df.mean().replace(0, np.nan)
cv = cv.replace([np.inf, -np.inf], np.nan)
cv_valid = cv.dropna()

plt.figure(figsize=(12,6))
sns.barplot(x=cv_valid.sort_values().index, y=cv_valid.sort_values().values, palette="magma")
plt.xticks(rotation=90)
plt.ylabel("CV (标准差/均值)")
plt.title("各变量变异系数", weight="bold")
plt.tight_layout()
plt.savefig(f"{out_dir}/cv.png", dpi=300)
plt.close()

# ===============================
# 4. KS 检验分布图（选前6个变量，带 NaN/常数保护）
# ===============================
sample_cols = df.columns[6:]
for col in sample_cols:
    data = df[col].dropna()
    mu, sigma = np.mean(data), np.std(data, ddof=0)
    
    plt.figure(figsize=(8,5))
    
    if sigma == 0 or len(data) < 5:  # 常数列或数据太少
        sns.histplot(data, bins=10, kde=False, color="gray", alpha=0.7)
        plt.title(f"{col} 为常数列或数据不足，无法进行KS检验", weight="bold")
    else:
        sns.kdeplot(data, label="实际分布", fill=True, color="steelblue", alpha=0.6)
        x = np.linspace(data.min(), data.max(), 200)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r--', linewidth=2, label="正态分布拟合")
        
        ks_stat, p_val = stats.kstest((data - mu)/sigma, 'norm')
        plt.title(f"{col} 分布对比 (KS={ks_stat:.3f}, p={p_val:.3f})", weight="bold")
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/ks_{col}.png", dpi=300, bbox_inches="tight")
    plt.close()

# ===============================
# 5. t-SNE 高维可视化
# ===============================
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], 
            c=X_tsne[:,0], cmap="Spectral", 
            s=40, alpha=0.7, edgecolors="none")
plt.title("t-SNE 高维数据可视化", weight="bold")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig(f"{out_dir}/tsne.png", dpi=300, bbox_inches="tight")
plt.close()

# ===============================
# 6. 数据质量雷达图（封面，排除常数列）
# ===============================

# 只选择标准差不为0的列计算 KS
valid_cols = [c for c in sample_cols if df[c].std(ddof=0) != 0]

if valid_cols:
    ks_values = []
    for c in valid_cols:
        data = df[c].dropna()
        mu, sigma = data.mean(), data.std(ddof=0)
        ks_stat, _ = stats.kstest((data - mu)/sigma, 'norm')
        ks_values.append(ks_stat)
    ks_score = 1 - np.mean(ks_values)  # 越接近1越好
else:
    ks_score = np.nan  # 如果全是常数列

# 构建质量指标
quality_metrics = {
    "均值稳定性": (1 / cv_valid.mean()),  # CV越小越好
    "信噪比": snr.mean(),
    "PCA信息保真度": cum_explained[1],  # 前两主成分解释度
    "正态性KS检验": ks_score,
    "高维可分性": np.std(X_tsne[:,0]) + np.std(X_tsne[:,1])  # t-SNE分布的离散度
}

# 去掉 NaN 值（例如 KS 指标全是常数列）
quality_metrics = {k: v for k, v in quality_metrics.items() if not np.isnan(v)}

# 雷达图绘制
labels = list(quality_metrics.keys())
stats_vals = list(quality_metrics.values())
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
stats_vals += stats_vals[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
ax.plot(angles, stats_vals, 'o-', linewidth=2, color="tab:blue")
ax.fill(angles, stats_vals, alpha=0.25, color="tab:blue")
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_title("数据质量综合雷达图", weight="bold", size=16, pad=20)
plt.tight_layout()
plt.savefig(f"{out_dir}/quality_radar.png", dpi=300, bbox_inches="tight")
plt.close()


print("✅ 数据质量图表已生成：PCA, SNR, CV, KS分布, t-SNE")