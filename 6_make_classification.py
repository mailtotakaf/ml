import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, load_iris

# データセットの生成
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, n_classes=2, random_state=42)

# # データセットの読み込み（ここではIrisデータセットを使用）
# iris = load_iris()
# X = iris.data  # 特徴量
# y = iris.target  # ラベル


# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの構築
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, input_dim=20, activation='relu'),  # 入力層（20個の入力特徴）
    tf.keras.layers.Dense(8, activation='relu'),  # 隠れ層
    tf.keras.layers.Dense(1, activation='sigmoid')  # 出力層（1つのユニット、2クラス分類のためシグモイド関数）
])

# モデルのコンパイル
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# モデルの学習
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# モデルの評価
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# 新しいデータに対する予測
new_data = np.random.rand(1, 20)  # 20次元の新しいデータ
prediction = model.predict(new_data)
print(f'Prediction: {prediction}')
