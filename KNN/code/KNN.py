import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

## 把encoding完的資料用KNN值補起來 

# 讀取CSV檔案
data = pd.read_csv(r"C:\Users\Hank\BoyorGirl\BoyorGirl\test_encoding.csv")

# 將 '#NUM!' 替換為 NaN
# data.replace('#NUM!', np.nan, inplace=True)

# 建立KNNImputer物件
imputer = KNNImputer(n_neighbors=5)  # 設定k=5,可以根據需求調整

# 選擇需要補值的欄位
columns_to_impute = ['star_sign', 'phone_os', 'height', 'weight', 'sleepiness', 'iq', 'fb_friends', 'yt']

# 對選定的欄位進行補值
data[columns_to_impute] = imputer.fit_transform(data[columns_to_impute])

# # 輸出補值後的資料
# print(data)

# 將處理過的資料輸出到新的 CSV 檔案
data.to_csv(r"C:\Users\Hank\BoyorGirl\BoyorGirl\test_KNN.csv", index=False)

