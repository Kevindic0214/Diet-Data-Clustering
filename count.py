import pandas as pd

# 1. 載入資料集
# 假設您的資料集為 'dietary_records.csv'
df = pd.read_csv('diet_data.csv')

# 2. 定義水果和蔬菜的特徵列表
fruits = ['Stewed', 'Prune', 'Dried', 'Mixed', 'Apple', 'Banana', 'Berry',
          'Cherry', 'Grapefruit', 'Grape', 'Mango', 'Melon', 'Orange',
          'Satsuma', 'Peach_nectarine', 'Pear', 'Pineapple', 'Plum', 'Other']

vegetables = ['Bakedbean', 'Pulses', 'Mixedvegetable', 'Vegetablepieces',
              'Coleslaw', 'Sidesalad', 'Avocado', 'Broadbean', 'Greenbean',
              'Beetroot', 'Broccoli', 'Butternutsquash', 'Cabbage_kale',
              'Carrot', 'Cauliflower', 'Celery', 'Courgette', 'Cucumber',
              'Garlic', 'Leek', 'Lettuce', 'Mushroom', 'Onion', 'Parsnip',
              'Pea', 'Sweetpepper', 'Spinach', 'Sprouts', 'Sweetcorn',
              'Freshtomato', 'Tinnedtomato', 'Turnip_swede', 'Watercress',
              'Othervegetables']

# 3. 定義各類別的數值範圍
fruit_servings = [0.5, 1, 2, 3, 4]
vegetable_servings = [0.25, 0.5, 1, 2, 3]

# 4. 建立一個空的字典來儲存結果
result = {}

# 5. 計算水果類別
for fruit in fruits:
    if fruit in df.columns:
        counts = df[fruit].value_counts().reindex(fruit_servings, fill_value=0)
        result[fruit] = counts

# 6. 計算蔬菜類別
for veg in vegetables:
    if veg in df.columns:
        counts = df[veg].value_counts().reindex(vegetable_servings, fill_value=0)
        result[veg] = counts

# 7. 將結果轉換為 DataFrame
result_df = pd.DataFrame(result).transpose()

# 8. 重新命名索引以清楚顯示類別數值
result_df = result_df.rename_axis('特徵').reset_index()

# 9. 顯示結果
print(result_df)

# 10. 若需匯出為 Excel 檔案
result_df.to_excel('feature_value_counts.xlsx', index=False)
