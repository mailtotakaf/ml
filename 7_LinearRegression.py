import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# データの生成 (例として簡単な線形データ)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # 説明変数
y = 4 + 3 * X + np.random.randn(100, 1)  # 目的変数

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 線形回帰モデルの作成と学習
model = LinearRegression()
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# 結果の表示
print(f"回帰係数: {model.coef_}")
print(f"切片: {model.intercept_}")
print(f"平均二乗誤差: {mean_squared_error(y_test, y_pred)}")

# グラフ表示
plt.scatter(X_test, y_test, color='black', label='Actual')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()
