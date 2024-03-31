import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
import numpy as np

# 加载数据
df = pd.read_csv(r'C:\AI專案\InformationData\Boys and girls\BoyorGirl\HotDeck\data\AfterHotDeck\TrainHotDeck_withBMI.csv')  # 请替换成你的文件路径

# 假设df是你的DataFrame，并且所有列都是数值型数据
features = df.columns

# 假设df是你的原始DataFrame
# 使用隔离森林找到异常值并标记
iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
preds = iso_forest.fit_predict(df[features])
df['outlier'] = preds == -1

# 对于被标记为异常值的行，我们可以先将它们转换为NaN
for feature in features:
    df.loc[df['outlier'] == True, feature] = np.nan

# 现在使用imputer来填充NaN值
imputer = SimpleImputer(strategy='median')
df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 确保不再需要'outlier'列，可以将其删除
df_filled.drop(['outlier'], axis=1, inplace=True)

# 保存处理过的DataFrame到新的CSV文件
df_filled.to_csv('processed_dataset.csv', index=False)


