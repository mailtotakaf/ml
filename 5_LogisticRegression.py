# 必要なライブラリのインポート
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# データの読み込み
iris = load_iris()

# 特徴量とターゲットの設定（2つのクラスに絞る）
X = iris.data[:100, :]  # 0~99行のデータを使用
y = iris.target[:100]   # 0と1のラベルを使用（2つのクラス）

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ロジスティック回帰モデルの作成
model = LogisticRegression()

# モデルの訓練
model.fit(X_train, y_train)

# テストデータで予測
y_pred = model.predict(X_test)

# 精度の計算
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 混合行列の作成
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('y_pred')
plt.ylabel('y_test')
plt.title('Confusion Matrix')
plt.show()