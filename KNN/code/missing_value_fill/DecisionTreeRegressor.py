import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer

# 讀取 CSV 文件
data = pd.read_csv(r"C:\Users\Hank\BoyorGirl\BoyorGirl\KNN\dataset\path_to_your_file_modified.csv")

# 需要填補缺失值的列
columns_to_impute = ['star_sign', 'phone_os', 'height', 'weight', 'sleepiness', 'iq', 'fb_friends', 'yt']

# 移除 self_intro 列
data_for_imputation = data.drop(columns=['self_intro'])

# 對每一列進行填充
for column in columns_to_impute:
    # 將數據拆分為有缺失值和無缺失值兩部分
    known = data_for_imputation[data_for_imputation[column].notnull()]
    unknown = data_for_imputation[data_for_imputation[column].isnull()]
    
    # 提取特徵和標籤
    X_known = known.drop(columns_to_impute, axis=1)
    y_known = known[column]
    X_unknown = unknown.drop(columns_to_impute, axis=1)

    # 將無效值替換為 NaN
    y_known = y_known.replace('#NUM!', np.nan)
    
    # 檢查 X_unknown 是否為空
    if X_unknown.shape[0] > 0:
        # 使用 DecisionTreeRegressor 進行填充
        imputer = DecisionTreeRegressor()
        imputer.fit(X_known, y_known)
        filled_values = imputer.predict(X_unknown)
    else:
        # 如果 X_unknown 為空，則不對該列進行任何操作
        filled_values = []

    # 將填充後的值放回到原始數據中
    data.loc[data[column].isnull(), column] = filled_values

# 將處理過的數據輸出到新的 CSV 文件
data.to_csv(r"C:\Users\Hank\BoyorGirl\BoyorGirl\KNN\dataset\train_DecisionTreeRegressor_imputed.csv", index=False)