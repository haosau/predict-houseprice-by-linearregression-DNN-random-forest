# 开发时间 2023/12/24 11:16
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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
data_encoded['total_rooms']=data_encoded['Bedrooms']+data_encoded['Bathrooms']
print(data_encoded.head())
#抽样
data_encoded=data_encoded.sample(frac=0.1,random_state=42)
# 选择特征和目标变量
features = data_encoded[['SquareFeet',  'total_rooms', 'HouseAge', 'Neighborhood_Suburb', 'Neighborhood_Urban']]
target = data_encoded['Price']
# 使用 Min-Max 归一化对特征进行缩放
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)
# 数据拆分
X_train, X_test, y_train, y_test = train_test_split(features_normalized, target, test_size=0.2, random_state=42)

# 创建并训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)


r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2}")
# 计算均方根误差 (RMSE)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse}")

# 打印目标变量 Price 的平均值
average_price = y_test.mean()
print(f"Average Price: {average_price}")


# 假设 model 是一个已经训练好的线性回归模型
intercept = model.intercept_
coefficients = model.coef_
feature_names = features.columns  # 假设 features 是你的特征矩阵

# 输出回归方程
equation = f"Regression Equation: y = {intercept}"
for feature, coef in zip(feature_names, coefficients):
    equation += f" + {coef:.4f} * {feature}"

print(equation)



# 绘制预测值与实际值的散点图
plt.scatter(y_test, y_pred, color='blue', label='Actual vs Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2, label='Perfect Prediction')

# 添加标签和标题
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot of Actual vs Predicted Values')

# 添加图例
plt.legend()

# 显示图形
plt.show()



# 计算残差
residuals = y_test - y_pred

# 绘制残差图
plt.scatter(y_pred, residuals, color='blue', label='Residuals')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Residual Line')

# 添加标签和标题
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot')

# 添加图例
plt.legend()

# 显示图形
plt.show()








