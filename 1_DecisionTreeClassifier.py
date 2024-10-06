# 必要なライブラリをインポート
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# データセットの読み込み（ここではIrisデータセットを使用）
iris = load_iris()
X = iris.data  # 特徴量
y = iris.target  # ラベル

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 決定木モデルの作成
clf = DecisionTreeClassifier()

# モデルの訓練
clf.fit(X_train, y_train)

# テストデータを使って予測
y_pred = clf.predict(X_test)

# モデルの精度を評価
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 予測結果を表示
print("予測結果:", y_pred)
print("実際のラベル:", y_test)

# 混合行列の作成
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('y_pred')
plt.ylabel('y_test')
plt.title('Confusion Matrix')
plt.show()