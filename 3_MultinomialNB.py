import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# データセットの作成（データ少ないと精度が低い）
reviews = [
    "I love this movie, it's fantastic!",
    "This film was terrible, I hated it.",
    "What a great film! I really enjoyed it.",
    "Worst movie ever, don't waste your time.",
    "Absolutely wonderful, I would watch it again.",
    "It's a boring movie, not recommended.",
    "What's that's mean?",
    "You are welcomw!",
    "Please kill it.",
    "Fuck you.",
    "You don't mind.",
    "Why are they shit me?",
    "hate me.",
    "Why don't you kill me?"
]

# ラベルの作成 (1: ポジティブ, 0: ネガティブ)
labels = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]

# # 分割のバイアスを減らすため（微妙）
# # データフレームの作成
# data = pd.DataFrame({'review': reviews, 'label': labels})

# # データをシャッフル
# data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# # シャッフルされたデータを表示
# print("Shuffled Data:")
# print(data)

# データの分割
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.33, random_state=42)

# テキストデータをベクトル化（特徴抽出方法）
# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# ナイーブベイズ分類器の作成
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# テストデータで予測
y_pred = classifier.predict(X_test_vectorized)

# 精度を表示
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 予測結果の表示
for review, prediction in zip(X_test, y_pred):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f'Review: "{review}" - Predicted Sentiment: {sentiment}')
