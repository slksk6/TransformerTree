import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 定义商品大类列表
product_categories = ['牛产品', '水产品', '鸡产品', '羊产品', '猪产品']

# 定义商品大类特征字典
category_features = {
    "牛产品": ['供应商ID', '厂号', '品种', '重量'],
    "水产品": ['等级', '申报类别', '切割方式'],
    "鸡产品": ['含肉率', '饲养方式', '原产地'],
    "羊产品": ['供应商ID', '等级', '申报类别'],
    "猪产品": ['供应商ID', '原产地']
}
all_results = []  # 创建一个列表来存储每个商品大类的结果

for category in product_categories:
    all_dfs = []
    df = pd.read_excel(f"out_{category}.xlsx")


    # 删除包含缺失值的行
    df = df.dropna()

    df_1 = df
    df_train = df_1
    X_train = df_train[category_features.get(category, [])].values
    Y_train = df_train['单价(原币)']

    # 读取测试集数据
    df_test = df_1.iloc[int(len(df_1) * 0.8):]
    X_test = df_test[category_features.get(category, [])].values
    Y_test = df_test['单价(原币)']

    # 决策树建模预测
    regressor = DecisionTreeRegressor(random_state=0).fit(X_train, Y_train)
    y_pred = regressor.predict(X_test)

    # 计算均方误差（MSE）
    mse = mean_squared_error(Y_test, y_pred)
    print(f"{category}均方误差（MSE）: {mse}")

    # 计算平均绝对误差（MAE）
    mae = mean_absolute_error(Y_test, y_pred)
    print(f"{category}平均绝对误差（MAE）: {mae}")



    # 可视化部分
    sns.set(font_scale=1.2)
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rc('font', size=14)

    # 取前50条数据进行对比
    num_samples = min(50, len(X_test))  # 确保不超过实际数据长度
    plt.plot(list(range(0, num_samples)), Y_test[:num_samples], marker='o', label='真实值')
    plt.plot(list(range(0, num_samples)), y_pred[:num_samples], marker='*', label='预测值')
    plt.legend()
    plt.title(f'{category}价格预测值与真实值的对比')
    plt.show()

    x_pred = pd.read_excel(f'output_{category}.xlsx')
    x_pred = x_pred[category_features.get(category, [])].values
    y_pred1 = regressor.predict(x_pred)

    # 将结果存储到列表中，并添加商品大类
    data = {
        '单价': y_pred1,
        '商品大类': [category] * len(y_pred1)  # 添加商品大类列
    }
    df_output = pd.DataFrame(data)
    all_results.append(df_output)  # 将当前商品大类的结果添加到列表中

# 将所有结果合并为一个 DataFrame
final_output = pd.concat(all_results, ignore_index=True)

# 将合并后的 DataFrame 保存到 Excel 文件
final_output.to_excel('result_combined.xlsx', index=False, engine='openpyxl')


