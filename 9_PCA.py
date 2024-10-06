import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd

# データセットのロード
iris = load_iris()
X = iris.data  # 特徴量
y = iris.target  # ラベル

# PCAの適用（2次元に削減）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 結果をデータフレームに変換
df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df['target'] = y

# 主成分の可視化
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
for i in range(3):
    subset = df[df['target'] == i]
    plt.scatter(subset['PC1'], subset['PC2'], color=colors[i], label=iris.target_names[i])
    
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.title('PCA of Iris Dataset')
plt.show()
