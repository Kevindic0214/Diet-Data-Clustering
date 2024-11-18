import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from minisom import MiniSom
import logging
import time

# 設定 logging 配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def read_data(file_path):
    """讀取數據文件"""
    try:
        data = pd.read_csv(file_path)
        logging.info(f"成功讀取數據檔案：{file_path}")
        return data
    except Exception as e:
        logging.error(f"無法讀取數據檔案：{file_path}, 錯誤：{e}")
        raise

def preprocess_data(data):
    """清理和處理缺失值"""
    if 'ID' in data.columns:
        data = data.drop(columns=['ID'])
        logging.info("成功刪除 ID 列")
    data = data.fillna(0)
    logging.info("缺失值已填補為 0")
    return data

def standardize_data(data):
    """標準化數據"""
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    logging.info("數據已標準化")
    return data_scaled

def train_som(data_scaled, x=10, y=10, sigma=1.0, learning_rate=0.5, iterations=100):
    """訓練 SOM"""
    logging.info("開始使用 SOM 進行降維")
    try:
        som = MiniSom(x=x, y=y, input_len=data_scaled.shape[1], sigma=sigma, learning_rate=learning_rate)
        som.train_random(data_scaled, iterations)
        logging.info("SOM 訓練完成")
        return som
    except Exception as e:
        logging.error(f"SOM 訓練失敗：{e}")
        raise

def find_best_k(som_labels, k_range):
    """選擇最佳 K 值"""
    sil_scores = []
    logging.info("開始選擇最佳 K 值")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans_labels = kmeans.fit_predict(som_labels)
        sil_score = silhouette_score(som_labels, kmeans_labels)
        sil_scores.append(sil_score)
        logging.debug(f"K={k}, 輪廓係數={sil_score}")
    best_k = k_range[sil_scores.index(max(sil_scores))]
    best_score = max(sil_scores)
    logging.info(f"最佳的簇數 K：{best_k}")
    logging.info(f"對應的輪廓係數：{best_score}")
    return best_k, best_score

def visualize_results(som_labels, kmeans_labels, best_k, cluster_centers=None):
    """可視化結果"""
    logging.info("開始可視化 SOM + K-means 聚類結果")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(som_labels[:, 0], som_labels[:, 1], c=kmeans_labels, cmap='viridis', s=50)
    ax.set_xlabel('SOM Grid X')
    ax.set_ylabel('SOM Grid Y')
    plt.title(f'SOM + K-means Clustering (Best K={best_k})')
    if cluster_centers is not None:
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200, marker='x', label='Cluster Centers')
        ax.legend()
    plt.colorbar(scatter)
    plt.savefig('som_kmeans_clustering.png', dpi=300)  # 儲存圖像
    plt.show()
    logging.info("可視化完成，結果已保存為 som_kmeans_clustering.png")

if __name__ == "__main__":
    start_time = time.time()
    logging.info("程式開始執行")

    # 讀取數據
    file_path = './diet_data_cleaned.csv'
    data = read_data(file_path)
    data = preprocess_data(data)
    data_scaled = standardize_data(data)

    # 將數據轉換為 PyTorch Tensor 並移到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_gpu = torch.tensor(data_scaled, dtype=torch.float32).to(device)
    logging.info(f"數據已轉換為 PyTorch Tensor，並移到設備：{device}")
    if torch.cuda.is_available():
        logging.info(f"可用 GPU：{torch.cuda.get_device_name(0)}")
    else:
        logging.warning("未偵測到 GPU，將使用 CPU 執行")

    # SOM 降維
    som = train_som(data_gpu.cpu().numpy())
    som_labels = np.array([som.winner(x) for x in data_gpu.cpu().numpy()])
    logging.info("成功提取 SOM 網格位置")

    # 選擇最佳 K 值
    k_range = range(2, 54)
    best_k, best_score = find_best_k(som_labels, k_range)

    # 使用最佳 K 值進行 K-means 聚類
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    kmeans_labels = kmeans.fit_predict(som_labels)
    cluster_centers = kmeans.cluster_centers_
    logging.info(f"使用最佳 K 值（K={best_k}）完成 K-means 聚類")

    # 可視化
    visualize_results(som_labels, kmeans_labels, best_k, cluster_centers)

    end_time = time.time()
    logging.info(f"程式總執行時間：{end_time - start_time:.2f} 秒")