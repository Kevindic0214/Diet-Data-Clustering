import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from itertools import product
import logging

# 設置 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. 讀取與清理資料
logging.info('讀取資料集')
diet_data_raw = pd.read_csv('diet_data_cleaned.csv')
logging.info(f'資料集包含 {diet_data_raw.shape[0]} 筆樣本和 {diet_data_raw.shape[1]} 個特徵')

# 移除 'ID' 欄位（如果存在）
if 'ID' in diet_data_raw.columns:
    diet_data_cleaned = diet_data_raw.drop(columns=['ID'])
else:
    diet_data_cleaned = diet_data_raw.copy()

# 填補缺失值
diet_data_cleaned = diet_data_cleaned.dropna(how='all').fillna(0)

# 2. 新增分類標籤
logging.info('新增分類標籤')
category_map = {
    # Fruit categories
    'Stewed': 'Other fruits',
    'Prune': 'Stone fruits',
    'Dried': 'Dried fruits',
    'Mixed': 'Mixed fruits',
    'Apple': 'Temperate fruits',
    'Banana': 'Tropical fruits',
    'Berry': 'Berries',
    'Cherry': 'Stone fruits',
    'Grapefruit': 'Citrus fruits',
    'Grape': 'Berries',
    'Mango': 'Tropical fruits',
    'Melon': 'Melons',
    'Orange': 'Citrus fruits',
    'Satsuma': 'Citrus fruits',
    'Peach_nectarine': 'Stone fruits',
    'Pear': 'Temperate fruits',
    'Pineapple': 'Tropical fruits',
    'Plum': 'Stone fruits',
    'Other': 'Other fruits',
    # Vegetable categories
    'Bakedbean': 'Legumes',
    'Pulses': 'Legumes',
    'Mixedvegetable': 'Mixed vegetables',
    'Vegetablepieces': 'Other vegetables',
    'Coleslaw': 'Cruciferous vegetables',
    'Sidesalad': 'Leafy vegetables',
    'Avocado': 'Fruit vegetables',
    'Broadbean': 'Legumes',
    'Greenbean': 'Legumes',
    'Beetroot': 'Root vegetables',
    'Broccoli': 'Cruciferous vegetables',
    'Butternutsquash': 'Cucurbitaceae vegetables',
    'Cabbage_kale': 'Leafy vegetables',
    'Carrot': 'Root vegetables',
    'Cauliflower': 'Cruciferous vegetables',
    'Celery': 'Leafy vegetables',
    'Courgette': 'Fruit vegetables',
    'Cucumber': 'Fruit vegetables', 
    'Garlic': 'Other vegetables',
    'Leek': 'Leafy vegetables',
    'Lettuce': 'Leafy vegetables',
    'Mushroom': 'Fungi',
    'Onion': 'Other vegetables',
    'Parsnip': 'Root vegetables',
    'Pea': 'Legumes',
    'Sweetpepper': 'Fruit vegetables',
    'Spinach': 'Leafy vegetables',
    'Sprouts': 'Cruciferous vegetables',
    'Sweetcorn': 'Cereal vegetables',
    'Freshtomato': 'Fruit vegetables',
    'Tinnedtomato': 'Fruit vegetables',
    'Turnip_swede': 'Root vegetables',
    'Watercress': 'Leafy vegetables',
    'Othervegetables': 'Other vegetables'
}
food_columns = list(category_map.keys())
missing_columns = set(food_columns) - set(diet_data_cleaned.columns)
if missing_columns:
    logging.warning(f'資料集中缺少以下食物列: {missing_columns}')

# 計算每個分類的總攝入量並新增分類
for category in set(category_map.values()):
    relevant_columns = [col for col, cat in category_map.items() if cat == category and col in diet_data_cleaned.columns]
    if relevant_columns:
        diet_data_cleaned[category] = diet_data_cleaned[relevant_columns].sum(axis=1)

diet_data_cleaned['Top_Category'] = diet_data_cleaned[list(set(category_map.values()))].idxmax(axis=1)

# 編碼分類標籤
label_encoder = LabelEncoder()
diet_data_cleaned['Category_Code'] = label_encoder.fit_transform(diet_data_cleaned['Top_Category'])

# 3. UMAP 降維
logging.info('準備進行 UMAP 降維')
neighbors_values = [10, 15, 20, 50, 100]
min_dist_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9]
umap_output_dir = 'umap_results'
os.makedirs(umap_output_dir, exist_ok=True)

for neighbors, min_dist in product(neighbors_values, min_dist_values):
    logging.info(f'處理參數組合: n_neighbors={neighbors}, min_dist={min_dist}')
    umap_model = umap.UMAP(
        n_neighbors=neighbors, 
        min_dist=min_dist, 
        n_components=2,
        metric='euclidean',
        random_state=42,
        verbose=True
    )
    # 直接使用清理後的數值型資料進行 UMAP 降維
    numerical_data = diet_data_cleaned.select_dtypes(include=[np.number]).drop(columns=['Category_Code'])
    umap_embedding = umap_model.fit_transform(numerical_data)
    
    # 可視化 UMAP 結果
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        umap_embedding[:, 0], umap_embedding[:, 1],
        c=diet_data_cleaned['Category_Code'],
        cmap='tab20',
        s=1,
        alpha=0.6
    )
    plt.title(f'UMAP Visualization (n_neighbors={neighbors}, min_dist={min_dist})')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    
    # 添加分類圖例
    legend_handles = [mpatches.Patch(color=scatter.cmap(scatter.norm(i)), label=cls)
                      for i, cls in enumerate(label_encoder.classes_)]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', title='Categories')
    
    # 保存圖片
    filename = f'umap_neighbors_{neighbors}_dist_{min_dist}.png'
    filepath = os.path.join(umap_output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    logging.info(f'UMAP 圖片已保存至 {filepath}')
