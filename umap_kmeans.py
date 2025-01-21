import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 設置 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. 讀取資料集
logging.info('讀取資料集')
data = pd.read_csv('diet_data_cleaned.csv')

# 刪除 'ID' 欄位
data = data.drop(columns=['ID'])

# 2. 檢查資料並進行預處理
logging.info('檢查資料並進行預處理')

# 確保所有特徵都是數值型別
numeric_columns = data.select_dtypes(include=[np.number]).columns
data_numeric = data[numeric_columns]
logging.info(f'數值特徵的數量: {len(numeric_columns)}')

# 3. 將資料轉換為 NumPy 陣列並進行標準化
logging.info('資料標準化')
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# 4. 設定固定的超參數值
n_neighbors = 15
min_dist = 0.001

# 5. 建立 UMAP 模型並執行降維
logging.info(f'使用固定參數進行處理: n_neighbors={n_neighbors}, min_dist={min_dist}')
reducer = umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    n_components=2,
    metric='euclidean',
    random_state=42,
    verbose=True
)
embedding = reducer.fit_transform(data_scaled)
logging.info(f'UMAP 降維完成，嵌入形狀: {embedding.shape}')

# 6. 執行 K-Means 聚類
n_clusters =  13 # 可以根據需要調整聚類數量
logging.info(f'執行 K-Means 聚類，n_clusters={n_clusters}')
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(embedding)
logging.info('K-Means 聚類完成')

# 7. 繪製散點圖並顯示聚類結果
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=labels,
    cmap='viridis',
    s=10,
    alpha=0.7
)
plt.gca().set_aspect('equal', 'datalim')
plt.title(f'UMAP with K-Means Clusters (n_neighbors={n_neighbors}, min_dist={min_dist}, n_clusters={n_clusters})')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.colorbar(scatter, ticks=range(n_clusters), label='Cluster')
plt.tight_layout()

# 8. 顯示圖形
plt.show()

logging.info('圖形已顯示')
