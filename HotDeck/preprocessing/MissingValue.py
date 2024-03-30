import pandas as pd

# 替换成你的 CSV 文件路径
file_path = r'C:\AI專案\InformationData\Boys and girls\BoyorGirl\path_to_your_csv_file_with_bmi.csv'
df = pd.read_csv(file_path)

# 计算并打印每列的缺失值数量
missing_values_count = df.isnull().sum()
print(missing_values_count)

# 如果你想获取数据中缺失值的总数
total_missing_values = df.isnull().sum().sum()
print(f'Total missing values in the dataset: {total_missing_values}')