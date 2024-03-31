import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
import numpy as np

# 加载数据
df = pd.read_csv(r'C:\AI專案\InformationData\Boys and girls\BoyorGirl\HotDeck\data\AfterHotDeck\TrainHotDeck_withBMI.csv')  # 请替换成你的文件路径

# 保存gender列的值
gender_column = df['gender'].copy()

# 假设我们要处理的特征列不包括'gender'和可能的其他不相关列
features_to_use = [col for col in df.columns if col not in ['gender', 'outlier']]  # 假设'outlier'是我们即将添加的列

# 使用隔离森林找到异常值并标记
iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
# 注意这里我们仅使用features_to_use里的特征进行fit_predict
preds = iso_forest.fit_predict(df[features_to_use])
df['outlier'] = preds == -1

# 对于被标记为异常值的行，将它们转换为NaN（除了'gender'）
for feature in features_to_use:
    df.loc[df['outlier'] == True, feature] = np.nan

# 使用KNNImputer来填充NaN值（除了'gender'和'outlier'）
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df[features_to_use]), columns=features_to_use)

# 将'gender'列的值重新赋值回去
df_imputed['gender'] = gender_column

# 如果'outlier'列不再需要，也可以在此步骤删除
df_imputed.drop(['outlier'], axis=1, inplace=True, errors='ignore')

# 您可能还希望重新排序列，以确保'gender'列在所期望的位置
columns_order = ['gender'] + [col for col in df_imputed.columns if col != 'gender']
df_imputed = df_imputed[columns_order]

# 保存处理过的DataFrame到新的CSV文件
df_imputed.to_csv('processed_dataset_with_KNNImputer.csv', index=False)
