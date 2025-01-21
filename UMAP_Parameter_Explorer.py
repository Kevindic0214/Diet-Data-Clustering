import pandas as pd
import numpy as np
import umap
import matplotlib
matplotlib.use('Agg')  # 使用非互動後端
import matplotlib.pyplot as plt
from itertools import product
import os
from sklearn.preprocessing import LabelEncoder
import logging
import matplotlib.patches as mpatches

# 設置 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. 讀取資料集
logging.info('讀取資料集')
data = pd.read_csv('diet_data.csv')
logging.info(f'資料集包含 {data.shape[0]} 筆樣本和 {data.shape[1]} 個特徵')

# 刪除 'ID' 欄位
data_without_id = data.drop(columns=['ID'])

# 刪除完全缺失的行
data_cleaned = data_without_id.dropna(how='all').fillna(0)
logging.info(f'資料集的形狀: {data_cleaned.shape}')

# 將清理後的資料賦值回 data，以保持變數名稱一致
data = data_cleaned

# 2. 檢查資料並進行預處理
logging.info('檢查資料並進行預處理')

# 計算每行非零元素的比例
non_zero_ratio = (data_cleaned != 0).sum(axis=0) / data_cleaned.shape[0]

# 設定非零比例門檻值（例如 > 5%）
threshold = 0.05  # 調整此值根據需求
logging.info(f'設定非零比例門檻值為 {threshold*100}%')

# 篩選非零比例大於門檻值的特徵
filtered_features = non_zero_ratio[non_zero_ratio > threshold].index.tolist()
filtered_data = data_cleaned[filtered_features]

# 獲取被刪除的特徵名稱
deleted_features = non_zero_ratio[non_zero_ratio <= threshold].index.tolist()
print(f"\n被刪除的特徵（非零比例 <= {threshold*100}%）：\n{deleted_features}")

# 更新後的資料集
logging.info(f'保留非零比例 > {threshold*100}% 的特徵，更新後資料集形狀: {filtered_data.shape}')

# 確保所有特徵都是數值型別
numeric_columns = data.select_dtypes(include=[np.number]).columns
data_numeric = data[numeric_columns]
logging.info(f'數值特徵的數量: {len(numeric_columns)}')

# 3. （已移除）資料標準化
# logging.info('資料標準化')
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(data_numeric)
# logging.info('資料標準化完成')

# 將標準化後的數據轉換回 DataFrame，並保留列名
# data_scaled_df = pd.DataFrame(data_scaled, columns=numeric_columns)

# 4. 新增分類標籤
logging.info('新增分類標籤')

# 建立項目到分類的映射字典
category_mapping = {
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

# 提取所有食物項目的列名
item_columns = list(category_mapping.keys())

# 確保資料集中包含所有的食物項目列
missing_columns = set(item_columns) - set(data.columns)
if missing_columns:
    logging.warning(f'資料集中缺少以下列: {missing_columns}')

# 計算每個分類的總攝入量
logging.info('計算每個分類的總攝入量')
for category in set(category_mapping.values()):
    # 找出屬於該分類的項目
    items_in_category = [item for item, cat in category_mapping.items() if cat == category]
    # 確保這些項目在資料集中存在
    items_in_category = [item for item in items_in_category if item in data.columns]
    if not items_in_category:
        continue
    # 計算總和，並新增為新的列
    data[category] = data[items_in_category].sum(axis=1)

# 為每個參與者找到攝入量最高的分類
data['主要分類'] = data[list(set(category_mapping.values()))].idxmax(axis=1)

# 為分類建立數值編碼
le = LabelEncoder()
data['分類編碼'] = le.fit_transform(data['主要分類'])

# 建立分類到顏色的映射
cmap = plt.get_cmap('tab20')  # 可以選擇其他 colormap
colors = cmap(np.linspace(0, 1, len(le.classes_)))
color_dict = dict(zip(le.classes_, colors))

# 5. 準備資料進行 UMAP 降維
logging.info('準備資料進行 UMAP 降維')
# 使用未標準化的數據進行 UMAP 降維
data_for_umap = data_numeric

# 6. 定義 n_neighbors 和 min_dist 的取值列表
n_neighbors_list = [10, 15, 20, 50, 100]
min_dist_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9]

# 7. 創建存放圖片的資料夾
output_dir = 'umap_visualizations_with_labels_without_standardize_and_with_dropna'
os.makedirs(output_dir, exist_ok=True)
logging.info(f'創建存放圖片的資料夾: {output_dir}')

# 8. 迭代所有參數組合，進行 UMAP 降維和可視化
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
    embedding = reducer.fit_transform(data_for_umap)
    logging.info(f'UMAP dimensionality reduction completed, embedding shape: {embedding.shape}')

    # 繪製散點圖，使用分類顏色
    plt.figure(figsize=(15, 8))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=data['分類編碼'],
        cmap='tab20',
        s=1,
        alpha=0.5
    )
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(f'UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')

    # 添加圖例
    handles = []
    for i, cls in enumerate(le.classes_):
        handles.append(mpatches.Patch(color=color_dict[cls], label=cls))
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title='Category')

    plt.tight_layout()

    # 保存圖片
    filename = f'umap_n{n_neighbors}_d{min_dist}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()

    logging.info(f'已保存圖片: {filepath}')
