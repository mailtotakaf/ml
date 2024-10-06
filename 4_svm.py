import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Irisデータセットを読み込む
iris = datasets.load_iris()
X = iris.data[:, :2]  # 最初の2つの特徴量を使用
y = iris.target

# トレーニングデータとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# SVMモデルの作成
model = svm.SVC(kernel='linear')  # 線形カーネルを使用
model.fit(X_train, y_train)  # モデルをトレーニング

# テストデータで予測
y_pred = model.predict(X_test)

# 精度を評価
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 決定境界をプロットするための関数
def plot_decision_boundary(X, y, model):
    # プロットの範囲を設定
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # 決定境界を計算
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 決定境界をプロット
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.show()

# 決定境界をプロット
plot_decision_boundary(X, y, model)
