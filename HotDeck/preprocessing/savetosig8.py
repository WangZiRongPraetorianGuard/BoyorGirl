import pandas as pd

# 加载数据，假定是'utf-8'或其他编码，这需要你根据文件实际情况进行调整
data_csv_path = r"C:\AI專案\InformationData\Boys and girls\data\boy or girl 2024 train_missingValue.csv"
df = pd.read_csv(data_csv_path, encoding='utf-8')

# 重存文件为'utf-8-sig'编码
output_csv_path = r"C:\AI專案\InformationData\Boys and girls\data\boy or girl 2024 train_missingValue.csv"
df.to_csv(output_csv_path, encoding='utf-8-sig', index=False)
