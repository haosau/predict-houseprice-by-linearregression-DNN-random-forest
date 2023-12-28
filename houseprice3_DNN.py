# 开发时间 2023/12/25 16:24
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)#s设定一个随机种子数确保实验可重复
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# 创建一个列表来存储每个epoch的R-squared值
r_squared_values = []

# 定义一个自定义的R-squared评价指标函数
def r_squared(y_true, y_pred):
    return tf.py_function(r2_score, (y_true, y_pred), tf.double)


# 构建 DNN 模型
model = Sequential()

# 添加输入层和隐藏层
model.add(Dense(8, input_dim=6, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(4, activation='linear'))
# 添加输出层
model.add(Dense(1, activation='linear'))

# 编译模型
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[r_squared])

# 训练模型
history=model.fit(X_train, y_train, epochs=50,batch_size=32, validation_data=(X_test, y_test))
# 提取训练过程中的R-squared值
#r_squared_values = history.history['r_squared']
val_r_squared_values = history.history['val_r_squared'] # 提取验证集上的 R-squared 值

# 找到最大的R-squared值和对应的epoch
max_r_squared = max(val_r_squared_values)
max_epoch =val_r_squared_values.index(max_r_squared) + 1  # 加1是因为epoch是从1开始的

# 打印最大的 R-squared 值和对应的 epoch
print(f'Max val_r-squared: {max_r_squared} at epoch {max_epoch}')

# 绘制 Max R-squared 随着 epochs 的变化图
plt.plot(range(1, len(val_r_squared_values) + 1), val_r_squared_values)
plt.xlabel('Epochs')
plt.ylabel('val_r_squared_values')
plt.legend()
plt.show()
val_rmse = mean_squared_error(y_test, model.predict(X_test), squared=False)
print(f'Validation RMSE: {val_rmse}')
# 打印目标变量 Price 的平均值
average_price = y_test.mean()
print(f"Average Price: {average_price}")


