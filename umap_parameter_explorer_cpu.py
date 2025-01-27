import pandas as pd
import numpy as np
import umap
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端
import matplotlib.pyplot as plt
from itertools import product
import os
from sklearn.preprocessing import StandardScaler
import logging

# 設置 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. 讀取資料集
logging.info('讀取資料集')
data = pd.read_csv('diet_data.csv')
logging.info(f'資料集包含 {data.shape[0]} 筆樣本和 {data.shape[1]} 個特徵')

# 刪除 'ID' 欄位
data_without_id = data.drop(columns=['ID'])

# 刪除完全缺失的行
data_cleaned = data_without_id.dropna(how='all')

# 將剩餘的 NaN 補為 0
data_cleaned = data_cleaned.fillna(0)
logging.info(f'資料集的形狀: {data_cleaned.shape}')

# 2. 檢查資料並進行預處理
logging.info('檢查資料並進行預處理')

# 確保所有特徵都是數值型別
numeric_columns = data_cleaned.select_dtypes(include=[np.number]).columns
data_numeric = data_cleaned[numeric_columns]
logging.info(f'數值特徵的數量: {len(numeric_columns)}')

# 3. 資料標準化
logging.info('資料標準化')
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# 4. 定義 n_neighbors 和 min_dist 的取值列表
n_neighbors_list = [3, 5, 10, 25, 50, 100, 200]
min_dist_list = [0.1, 0.2, 0.3, 0.4, 0.5]

# 5. 創建存放圖片的資料夾
output_dir = 'umap_visualizations_standardized'
os.makedirs(output_dir, exist_ok=True)
logging.info(f'創建存放圖片的資料夾: {output_dir}')

# 6. 迭代所有參數組合
for n_neighbors, min_dist in product(n_neighbors_list, min_dist_list):
    logging.info(f'處理參數組合: n_neighbors={n_neighbors}, min_dist={min_dist}')
    # 建立 UMAP 模型
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, 
        min_dist=min_dist, 
        n_components=2,
        metric='euclidean',
        random_state=42,
        verbose=True
    )
    # 執行降維
    embedding = reducer.fit_transform(data_scaled)
    
    # 繪製散點圖
    plt.figure(figsize=(8, 6))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=1, 
        alpha=0.5
    )
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(f'UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.tight_layout()
    
    # 保存圖片
    filename = f'umap_n{n_neighbors}_d{min_dist}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    logging.info(f'已保存圖片: {filepath}')