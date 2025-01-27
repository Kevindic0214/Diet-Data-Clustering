import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import logging
import os
from itertools import product

# 設置 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. 讀取與清理資料
logging.info('讀取資料集')
diet_data_raw = pd.read_csv('diet_data_cleaned.csv')
logging.info(f'資料集包含 {diet_data_raw.shape[0]} 筆樣本和 {diet_data_raw.shape[1]} 個特徵')

# 移除 'ID' 欄位（如果存在）
if 'ID' in diet_data_raw.columns:
    diet_data_cleaned = diet_data_raw.drop(columns=['ID'])
    logging.info("已移除 'ID' 欄位")
else:
    diet_data_cleaned = diet_data_raw.copy()
    logging.info("'ID' 欄位不存在，跳過移除步驟")

# 2. UMAP 降維 - 固定超參數
logging.info('準備進行 UMAP 降維')
numeric_data = diet_data_cleaned.select_dtypes(include=[np.number])

umap_model = umap.UMAP(
    n_neighbors=10, 
    min_dist=0.5, 
    n_components=2,
    metric='euclidean',
    random_state=42,
    verbose=True
)

# 使用原始數值特徵作為 UMAP 的輸入
umap_embedding = umap_model.fit_transform(numeric_data)
logging.info('UMAP 降維完成')

# 3. 進行 DBSCAN 聚類並搜尋最佳參數
logging.info('開始 DBSCAN 聚類')

eps_values = [1.0, 3.0, 5.0, 7.0, 9.0]
min_samples_values = [10, 15, 20, 25]

best_sil_score = -1
best_eps = None
best_min_samples = None
best_labels = None

for eps, min_samples in product(eps_values, min_samples_values):
    logging.info(f"測試 DBSCAN 參數: eps={eps}, min_samples={min_samples}")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(umap_embedding)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_non_noise_points = np.count_nonzero(labels != -1)
    logging.info(f"DBSCAN 聚類結果：群組數量={n_clusters}，非噪音點數量={num_non_noise_points}")

    if n_clusters > 1 and num_non_noise_points > 1:
        sil_score = silhouette_score(umap_embedding[labels != -1], labels[labels != -1])
        logging.info(f"Silhouette Score：{sil_score}")

        if sil_score > best_sil_score:
            best_sil_score = sil_score
            best_eps = eps
            best_min_samples = min_samples
            best_labels = labels
    else:
        logging.info("無法計算 Silhouette Score，因為群組數量少於 2 或沒有足夠的非噪音點。")

if best_eps is not None and best_min_samples is not None:
    logging.info(f"最佳 DBSCAN 參數：eps={best_eps}, min_samples={best_min_samples}, Silhouette Score={best_sil_score}")
else:
    logging.warning("未找到合適的 DBSCAN 參數組合。")

# 4. 可視化 DBSCAN 聚類結果
if best_labels is not None:
    logging.info("可視化 DBSCAN 聚類結果")
    plt.figure(figsize=(12, 8))

    # 使用最佳的 DBSCAN 聚類結果進行可視化
    unique_labels = set(best_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        class_member_mask = (best_labels == k)
        if k == -1:
            col = 'black'
            label_name = 'Noise'
        else:
            label_name = f'Cluster {k}'

        xy = umap_embedding[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], label=label_name, edgecolors='k', s=20)

    plt.title(f'DBSCAN Clustering Results (eps={best_eps}, min_samples={best_min_samples})')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()

    # 保存 DBSCAN 聚類結果圖片
    filename = f'dbscan_clusters_eps_{best_eps}_min_samples_{best_min_samples}.png'
    filepath = os.path.join('dbscan_results', filename)
    os.makedirs('dbscan_results', exist_ok=True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    logging.info(f"DBSCAN 聚類結果已保存至 {filepath}")
else:
    logging.warning("由於未找到最佳的 DBSCAN 聚類結果，跳過可視化步驟。")
