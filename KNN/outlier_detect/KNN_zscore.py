import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.impute import KNNImputer

# 讀取資料集
file_path = r'C:\Users\Hank\BoyorGirl\BoyorGirl\KNN\dataset\path_to_your_file_modified.csv'
data = pd.read_csv(file_path)

# 将'#NUM!'替换为NaN
data.replace('#NUM!', np.nan, inplace=True)

# 将非数值类型的值替换为 NaN
data['yt'] = pd.to_numeric(data['yt'], errors='coerce')

# 打印 'yt' 列中的所有唯一值，确认所有非数值类型的值都已替换为 NaN
# print(data['yt'].unique())

# 設置 pandas 顯示格式
pd.set_option('display.float_format', lambda x: '%.1E' % x if abs(x) > 1000 else '%.0f' % x)

# 欄位不包含 'id' 和 'self_intro'
columns_to_check = [col for col in data.columns if col not in ['id', 'self_intro']]

# 初始化計數器
total_outliers = 0
outliers_gender_1 = 0
outliers_gender_2 = 0

# 找出有outlier的資料
outliers = []
for index, row in data.iterrows():
    is_outlier = False
    for i,col in enumerate(columns_to_check):
        try:
            # 如果該欄位是負值，則視為outlier
            if row[col] < 0:
                is_outlier = True
                break
        except Exception as e:
            print("Error occurred while processing column:", col)
            print("Error message:", str(e))

        # 如果該欄位的數值過大，則視為outlier
        if col in ['height'] and (row[col] > 250 or row[col] < 140):  # 可以自行調整數值範圍
            is_outlier = True
            break

        if col == 'height':
            # 根据性别设置身高的正常范围
            if row['gender'] == 1:  # 男性
                normal_range = (150, 220)
            else:  # 女性
                normal_range = (140, 180)
            # 如果身高超出正常范围，则视为outlier
            if row[col] < normal_range[0] or row[col] > normal_range[1]:
                is_outlier = True
                break
        
        if col in ['weight'] and (row[col] > 200 or row[col] < 30):  # 可以自行調整數值範圍
            is_outlier = True
            break

        if col == 'weight':
            # 根据性别设置体重的正常范围
            if row['gender'] == 1:  # 男性
                normal_range = (40, 150)
            else:  # 女性
                normal_range = (30, 80)
            # 如果体重超出正常范围，则视为outlier
            if row[col] < normal_range[0] or row[col] > normal_range[1]:
                is_outlier = True
                break

        if col in ['fb_friends'] and row[col] > 10000:  # 可以自行調整數值範圍
            is_outlier = True
            break
        
        try:
            if col in ['yt'] and row[col] > 5000:  # 可以自行調整數值範圍
                is_outlier = True
                break
        except Exception as e:
            print("Error occurred while processing column:", col)
            print("Error message:", str(e))
            print("Error occurred at row index:", i)

        if col in ['bmi'] and row[col] > 40:
            is_outlier = True
            break

        # 計算z-score
        z_score = zscore([row[col]])[0]
        # 如果z-score的絕對值大於3，則視為outlier
        if abs(z_score) > 3:
            is_outlier = True
            break

    if is_outlier:
        outliers.append(index)

        total_outliers += 1
        if row['gender'] == 1:
            outliers_gender_1 += 1
        elif row['gender'] == 2:
            outliers_gender_2 += 1

# 列印出有outlier的資料
if outliers:
    print("有outlier的資料:")
    print(data.loc[outliers])
else:
    print("沒有outlier的資料")

# 列印出outlier的總筆數和gender為1和2的outlier筆數
print("Outlier的總筆數:", total_outliers)
print("Gender為1的Outlier筆數:", outliers_gender_1)
print("Gender為2的Outlier筆數:", outliers_gender_2)

outliers_csv = pd.DataFrame(data.loc[outliers])

# 移除outlier值並進行KNN補值
def handle_outliers_and_impute(data, outliers_indices, gender):
    # 移除outlier值
    cleaned_data = data.drop(outliers_indices)

    # 取得指定性別的資料索引
    gender_indices = cleaned_data[cleaned_data['gender'] == gender].index.tolist()

    # 移除非數值型態的欄位
    numeric_cols = cleaned_data.select_dtypes(include=['number']).columns.tolist()
    cleaned_data = cleaned_data[numeric_cols]

    # 使用KNN補值
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = cleaned_data.copy()
    imputed_data.loc[gender_indices] = imputer.fit_transform(imputed_data.loc[gender_indices])

    return imputed_data

# 對gender為1的outliers進行處理
if outliers_gender_1 > 0:
    gender_1_outliers_indices = data[(data['gender'] == 1) & data.index.isin(outliers)].index
    data = handle_outliers_and_impute(data, gender_1_outliers_indices, 1)

# 對gender為2的outliers進行處理
if outliers_gender_2 > 0:
    gender_2_outliers_indices = data[(data['gender'] == 2) & data.index.isin(outliers)].index
    data = handle_outliers_and_impute(data, gender_2_outliers_indices, 2)

# 列印修正後的資料
print(data)

data.to_csv(r"C:\Users\Hank\BoyorGirl\BoyorGirl\KNN\outlier_detect\dataset_without_outlier_by_KNN.csv", index=False)