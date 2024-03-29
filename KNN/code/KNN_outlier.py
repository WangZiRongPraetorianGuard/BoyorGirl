import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer 

# 讀取CSV檔案
data = pd.read_csv(r"C:\Users\Hank\BoyorGirl\BoyorGirl\KNN_finish_file.csv")

# 移除id欄位
data = data.drop('id', axis=1)

# 檢查每個欄位的數據類型
data_types = data.dtypes

# # 輸出每個欄位的數據類型
# print(data_types)

# 對數值型變數檢測outlier
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    # 計算z-score
    z_scores = stats.zscore(data[col])
    
    # 找出絕對值z-score > 3的outlier
    outliers = data[col][(z_scores > 3) | (z_scores < -3) | (data[col] < 0)]
    
    # 繪製箱形圖檢視outlier
    plt.figure(figsize=(10, 6))
    plt.boxplot(data[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
    
    # 顯示outlier值
    print(f"Outliers in {col}:")
    print(outliers)
    
    # 詢問是否要刪除outlier
    remove_outliers = input(f"Do you want to remove outliers in {col}? (y/n) ")
    if remove_outliers.lower() == 'y':
        data = data[~(z_scores > 3) & ~(z_scores < -3) | (data[col] < 0)]

data.to_csv(r"C:\Users\Hank\BoyorGirl\BoyorGirl\KNN_without_outlier.csv", index=False)

