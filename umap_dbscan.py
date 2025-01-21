import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import logging
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os

# 設置 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. 讀取資料集
logging.info('讀取資料集')
data = pd.read_csv('diet_data_cleaned.csv')
logging.info(f'資料集包含 {data.shape[0]} 筆樣本和 {data.shape[1]} 個特徵')

# 刪除 'ID' 欄位
if 'ID' in data.columns:
    data = data.drop(columns=['ID'])
    logging.info("已刪除 'ID' 欄位")
else:
    logging.warning("'ID' 欄位不存在，無需刪除")

# 2. 檢查資料並進行預處理（如果需要）
logging.info('檢查資料並進行預處理')

# 確保所有特徵都是數值型別
numeric_columns = data.select_dtypes(include=[np.number]).columns
data_numeric = data[numeric_columns]
logging.info(f'數值特徵的數量: {len(numeric_columns)}')

# 3. 資料標準化
logging.info('進行資料標準化')
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)
logging.info('資料標準化完成')

# 5. 設定固定的 UMAP 超參數值
n_neighbors = 15
min_dist = 0.001

# 6. 建立 UMAP 模型並執行降維
logging.info(f'使用固定參數進行處理: n_neighbors={n_neighbors}, min_dist={min_dist}')
reducer = umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    n_components=2,
    random_state=42,
    verbose=True
)
embedding = reducer.fit_transform(data_scaled)
logging.info(f'UMAP 降維完成，嵌入形狀: {embedding.shape}')

# 7. 執行 DBSCAN 聚類
# 設定 DBSCAN 的參數
eps = 0.5         # 鄰域半徑
min_samples = 5   # 最小樣本數
logging.info(f'執行 DBSCAN 聚類，參數: eps={eps}, min_samples={min_samples}')
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(embedding)
logging.info('DBSCAN 聚類完成')

# 8. 繪製散點圖並顯示聚類結果
plt.figure(figsize=(10, 8))

# 設定顏色映射，將噪聲點（標籤為 -1）設為黑色
unique_labels = set(labels)
# 使用不同的色譜以應對不同數量的群集
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # 噪聲點
        col = 'k'
        label = 'Noise'
    else:
        label = f'Cluster {k} ({np.sum(labels == k)})'
    class_member_mask = (labels == k)
    plt.scatter(
        embedding[class_member_mask, 0],
        embedding[class_member_mask, 1],
        c=[col],
        label=label,
        s=10,
        alpha=0.7
    )

plt.gca().set_aspect('equal', 'datalim')
plt.title(f'UMAP with DBSCAN Clusters (n_neighbors={n_neighbors}, min_dist={min_dist})')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend(markerscale=2, fontsize='small')
plt.tight_layout()

# 9. 顯示圖形
plt.show()

logging.info('圖形已顯示')
