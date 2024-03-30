import pandas as pd
from sklearn.metrics.pairwise import nan_euclidean_distances
import numpy as np

# 确保 DataFrame 的引用列是正确的数据类型，例如浮点数
def preprocess_columns(df, reference_columns):
    for column in reference_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')  # 将无法转换的值设为 NaN
    return df

def hot_deck_imputation(df, target_column, reference_columns):
    # 预处理参考列
    df = preprocess_columns(df, reference_columns)
    
    incomplete_rows = df[df[target_column].isna()]
    complete_rows = df[df[target_column].notna()]
    
    for idx, incomplete_row in incomplete_rows.iterrows():
        distances = nan_euclidean_distances([incomplete_row[reference_columns]], 
                                            complete_rows[reference_columns])[0]
        
        # 检查是否所有距离值都是 NaN
        if np.isnan(distances).all():
            print(f"Skipping row {idx} due to all-NaN distances.")
            continue
        
        closest_idx = complete_rows.index[np.nanargmin(distances)]
        df.at[idx, target_column] = df.at[closest_idx, target_column]

    return df

file_path = r'C:\AI專案\InformationData\Boys and girls\BoyorGirl\HotDeck\data\encoding\encoding_train.csv'  # 你的 CSV 文件路径
df = pd.read_csv(file_path)
df.replace('#NUM!', np.nan, inplace=True)

# 迭代每个列进行热牌填充
for column in df.columns:
    if column == 'weight':  # 如果是你已经示例处理过的列，跳过或者将其放在循环外单独处理
        continue
    reference_columns = [col for col in df.columns if col != column]  # 使用除了当前目标列之外的所有列作为参考
    df = hot_deck_imputation(df, column, reference_columns)
# 假设 'weight' 列的缺失值没有被填充
# 重新执行填充操作，但这次只针对 'weight' 列
reference_columns = [col for col in df.columns if col != 'weight']  # 排除 'weight' 列
df = hot_deck_imputation(df, 'weight', reference_columns)

# 再次检查 'weight' 列的缺失值数量
print(df['weight'].isna().sum())

# 储存填充后的 DataFrame
df.to_csv('C:\AI專案\InformationData\Boys and girls\BoyorGirl\HotDeck\data\AfterHotDeck/testhotdeck.csv', index=False)
