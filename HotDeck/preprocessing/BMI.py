import pandas as pd

# 替换成你的 CSV 文件路径
file_path = r'C:\AI專案\InformationData\Boys and girls\BoyorGirl\HotDeck\data\AfterHotDeck\KNN_withBMI.csv'
df = pd.read_csv(file_path)

# 确保 'height' 列是以米为单位，'weight' 是以千克为单位
# 如果 'height' 列以厘米为单位，需要将其转换为米
df['height'] = df['height'] / 100  # 假设身高单位为厘米，需要转换为米

# 计算 BMI
df['bmi'] = df['weight'] / (df['height'] ** 2)

# 查看前几行数据，包括新的 BMI 列
print(df.head())

# （可选）保存处理后的 DataFrame 到新的 CSV 文件
df.to_csv('path_to_your_csv_file_with_bmi.csv', index=False)
