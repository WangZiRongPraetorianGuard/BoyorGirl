import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 讀取資料集
file_path = r'C:\Users\Hank\BoyorGirl\BoyorGirl\KNN\dataset\KNN_finish_file.csv'
data = pd.read_csv(file_path)

# 欄位列表（排除'id'和'self_intro'欄位）
columns_to_check = [col for col in data.columns if col not in ['id', 'self_intro']]

# 設定檢測離群值的門檻（例如：3倍的標準差）
threshold = 3

# 檢查每個欄位是否有離群值
for col in columns_to_check:
    # 計算欄位的平均值和標準差
    mean = data[col].mean()
    std_dev = data[col].std()
    
    # 計算離群值的門檻
    lower_threshold = mean - threshold * std_dev
    upper_threshold = mean + threshold * std_dev
    
    # 找出離群值
    outliers = data[(data[col] < lower_threshold) | (data[col] > upper_threshold)]
    
    # 如果有離群值，顯示相關訊息
    if not outliers.empty:
        print(f"在欄位 '{col}' 中發現離群值:")
        print(outliers)
        print("\n")
    
    # 繪製直方圖和箱形圖以視覺化資料分佈和離群值
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(data[col], bins=20, color='skyblue', edgecolor='black')
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    plt.boxplot(data[col])
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    
    plt.show()
