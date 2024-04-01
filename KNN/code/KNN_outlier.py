import pandas as pd
import numpy as np
from scipy import stats

# 读取CSV文件
data = pd.read_csv(r"C:\Users\Hank\BoyorGirl\BoyorGirl\KNN\dataset\path_to_your_file_modified.csv")

# 储存所有含有outlier的数据的索引
outlier_indices = []

# 对每个字段检测outlier
for index, row in data.iterrows():
    is_outlier = False
    for col in data.columns:
        if isinstance(row[col], (int, float)):
            # 计算z-score
            z_score = stats.zscore(data[col])
            # 找出绝对值z-score > 3或值<0的outlier的索引
            if np.abs(z_score[index]) > 3 or row[col] < 0:
                is_outlier = True
                break
    if is_outlier:
        outlier_indices.append(index)

# 列出所有含有outlier的数据
if outlier_indices:
    print("Data with outliers:")
    outlier_data = data.loc[outlier_indices]
    print(outlier_data)
else:
    print("No outliers found in the dataset.")
