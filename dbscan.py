# 匯入必要的庫
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("程式開始執行")

# 1. 讀取 CSV 資料
try:
    data = pd.read_csv('diet_data_cleaned.csv')
    logging.info("成功讀取 CSV 資料")
except Exception as e:
    logging.error(f"讀取 CSV 資料失敗: {e}")
    raise

# 2. 處理缺失值
data_filled = data.fillna(0)
logging.info(f"資料缺失值已填充為零，共處理 {data.isnull().sum().sum()} 個缺失值")

# 3. 資料標準化
features = data_filled.columns[1:]  # 假設第一列是 ID
X = data_filled[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
logging.info("資料已完成標準化處理")

# 4. 使用 PCA 進行降維
pca = PCA(n_components=0.90)
X_pca = pca.fit_transform(X_scaled)
logging.info(f"PCA 降維完成，保留特徵數量：{X_pca.shape[1]}")

# 5. 自動測試超參數，找到較佳的 eps 值
min_samples_range = [5, 7, 10]
eps_range = np.arange(5.0, 7.5, 0.5)

best_silhouette = -1
best_eps = None
best_min_samples = None
best_labels = None

logging.info("開始進行 DBSCAN 超參數測試")
for min_samples in min_samples_range:
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X_pca)
    distances, indices = neighbors_fit.kneighbors(X_pca)
    distances = np.sort(distances[:, min_samples - 1], axis=0)
    
    for eps_value in eps_range:
        dbscan = DBSCAN(eps=eps_value, min_samples=min_samples)
        labels = dbscan.fit_predict(X_pca)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters <= 1 or n_clusters >= len(X_pca):
            continue
        
        if n_clusters > 1 and np.count_nonzero(labels != -1) > 1:
            sil_score = silhouette_score(X_pca[labels != -1], labels[labels != -1])
            logging.info(f"測試結果 - min_samples: {min_samples}, eps: {eps_value}, 群組數量: {n_clusters}, Silhouette Score: {sil_score}")
            
            if sil_score > best_silhouette:
                best_silhouette = sil_score
                best_eps = eps_value
                best_min_samples = min_samples
                best_labels = labels.copy()

# 輸出最佳結果
if best_eps is not None and best_min_samples is not None:
    logging.info(f"最佳參數 - min_samples: {best_min_samples}, eps: {best_eps}, Silhouette Score: {best_silhouette}")
else:
    logging.warning("未找到適合的參數組合")

# 6. 使用最佳參數進行最終聚類
dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
labels = dbscan.fit_predict(X_pca)
logging.info("最終 DBSCAN 聚類完成")

# 7. 將群組標籤添加到原始資料中
data_filled['Cluster'] = labels
logging.info(f"群組標籤已添加到資料中，共有 {len(set(labels))} 個群組（包括噪音）")

# 8. 生成聚類細節
cluster_counts = data_filled['Cluster'].value_counts().sort_index()
logging.info("群組樣本數量統計完成")

cluster_profiles = data_filled[data_filled['Cluster'] != -1].groupby('Cluster').mean()
logging.info("群組特徵平均值計算完成")

# 9. 可視化聚類結果
logging.info("開始繪製聚類結果圖表")
plt.figure(figsize=(10, 6))
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    class_member_mask = (labels == k)
    if k == -1:
        col = 'black'
        label_name = '噪音點'
    else:
        label_name = f'群組 {k}'
    
    xy = X_pca[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=[col], label=label_name, edgecolors='k', s=20)

plt.title('DBSCAN 聚類結果')
plt.xlabel('主成分 1')
plt.ylabel('主成分 2')
plt.legend()
plt.show()

logging.info("程式執行完成")