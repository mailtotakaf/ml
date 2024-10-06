import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# サンプルデータを生成
np.random.seed(42)
X = np.random.rand(100, 2)

# K-means クラスタリングの実行
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# クラスタリング結果を取得
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 結果のプロット
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='x')
plt.title("K-means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
