# 开发时间 2023/12/25 15:43

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
# 读取 CSV 文件
file_path = r"D:\研究生\housing_price_dataset.csv"  # 替换成你实际的文件路径
data = pd.read_csv(file_path)
#print(data.head())
#print(data.describe())
data_encoded = pd.get_dummies(data, columns=['Neighborhood'], drop_first=True)
#print(data.head())
#print(data_encoded.isnull().sum())
#print(data_encoded.head())
# 计算房屋年龄
current_year = 2023  # 假设当前年份为2023
data_encoded['HouseAge'] = current_year - data_encoded['YearBuilt']


# 选择特征和目标变量
features = data_encoded[['SquareFeet', 'Bedrooms', 'Bathrooms', 'HouseAge', 'Neighborhood_Suburb', 'Neighborhood_Urban']]
target = data_encoded['Price']
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)#s设定一个随机种子数确保实验可重复
# 初始化随机森林模型
rf_model = RandomForestRegressor(n_estimators=30,max_depth=5, random_state=42)

# 训练模型
rf_model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred_rf = rf_model.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest R-squared: {r2_rf}")
# 计算均方根误差 (RMSE)
rmse = mean_squared_error(y_test, y_pred_rf, squared=False)
print(f"RMSE: {rmse}")

# 打印目标变量 Price 的平均值
average_price = y_test.mean()
print(f"Average Price: {average_price}")
