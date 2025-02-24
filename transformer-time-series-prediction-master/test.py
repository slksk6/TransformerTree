import pandas as pd

df_1 = pd.read_excel('data/features.xlsx')
df_2 = pd.read_excel('output.xlsx')

all_data = []  # 用于存储所有循环生成的数据
all_data.append(df_1)
all_data.append(df_2)

final_data = pd.concat(all_data, ignore_index=True)  # 忽略原始索引，重新生成新的连续整数索引
final_data.to_excel("data/features.xlsx", index=False, engine='openpyxl')   # 将最终结果保存到Excel文件中
