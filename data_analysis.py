import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# データの読み込み
data = pd.read_csv('ecommerce_sales_data.csv')

# 欠損値の処理
data.ffill(inplace=True)

# データ型の変換と異常値の処理
data['purchase_amount'] = pd.to_numeric(data['purchase_amount'], errors='coerce')
data.dropna(subset=['purchase_amount'], inplace=True)

# 売上の分布を確認
sns.histplot(data['purchase_amount'])
plt.show()

# 商品カテゴリと売上の関係
sns.barplot(x='category', y='purchase_amount', data=data)
plt.show()

# 文字列型を数値型に変換
data = pd.get_dummies(data, columns=['gender', 'category'])

# 日付データの処理
data['purchase_date'] = pd.to_datetime(data['purchase_date']).astype(int) / 10**9

# 特徴量と目的変数を定義
X = data.drop('purchase_amount', axis=1)
y = data['purchase_amount']

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# モデルの訓練
model = LinearRegression()
model.fit(X_train, y_train)

# モデルの評価
y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
