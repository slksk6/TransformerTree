import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 读取Excel表格
df = pd.read_excel('data/数据集.xlsx')

df['单价(原币)'] = df['单价(原币)'] * df['人民币汇率']
df['到达港'] = df['到达港'].astype(str)
df['等级'] = df['等级'].astype(str)
df['供应商ID'] = df['供应商ID'].astype(str)
le = LabelEncoder()
for column in df.columns:
    if column not in ['采购时间', '单价(原币)', '商品大类']:
        df[column] = le.fit_transform(df[column])
# 将日期列转换为日期时间格式
df['采购时间'] = pd.to_datetime(df['采购时间'])


# 根据年月分组，计算其他列的平均值
result = df.groupby('采购时间').mean()

# 将计算得到的平均值存储到新的表格中
result.to_excel('out.xlsx')
