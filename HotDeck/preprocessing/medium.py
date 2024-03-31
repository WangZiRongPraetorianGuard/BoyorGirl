import pandas as pd
import numpy as np

def fill_missing_values_with_median(df, target_column):
    median_value = df[target_column].median()
    df[target_column].fillna(median_value, inplace=True)
    return df
def ensure_numeric_columns(df, columns):
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    return df
def check_and_print_column_types(df):
    numeric_columns = []
    non_numeric_columns = []
    
    # 检查每列的数据类型
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric_columns.append(column)
        else:
            non_numeric_columns.append(column)
    
    # 打印数值型列和非数值型列的信息
    print(f"Numeric columns: {numeric_columns}")
    print(f"Non-numeric columns: {non_numeric_columns}")
    
    return numeric_columns, non_numeric_columns

file_path = r'C:\AI專案\InformationData\Boys and girls\BoyorGirl\HotDeck\data\encoding\encoding_train.csv'
df = pd.read_csv(file_path)
df = ensure_numeric_columns(df, df.columns)

df.replace('#NUM!', np.nan, inplace=True)

# 检测并打印列的数据类型信息
numeric_columns, non_numeric_columns = check_and_print_column_types(df)

for column in numeric_columns:
    df = fill_missing_values_with_median(df, column)

# 储存填充后的 DataFrame
output_path = r'C:\AI專案\InformationData\Boys and girls\BoyorGirl\HotDeck\data\AfterHotDeck\Train\Train_Medium.csv'
df.to_csv(output_path, index=False)
