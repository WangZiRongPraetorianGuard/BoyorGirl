import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.pipeline import Pipeline

# 加载数据
df = pd.read_csv(r'C:\AI專案\InformationData\Boys and girls\BoyorGirl\HotDeck\data\AfterHotDeck\KNN_withBMI.csv')

# 数据预处理
# 处理缺失值并转换性别标签
imputer = SimpleImputer(strategy='mean')
df_imputed = imputer.fit_transform(df.drop(columns=['gender']))
df['gender'] = df['gender'].apply(lambda x: 0 if x == 1 else 1)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_imputed)  # 标准化处理

X = df_scaled
y = df['gender'].values

# 创建包含预处理步骤的 XGBoost 模型管道
xgboost_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
])

# 应用交叉验证
scores = cross_val_score(xgboost_pipeline, X, y, cv=5)  # cv 参数表示交叉验证分割数目

# 打印每次交叉验证的准确率以及平均准确率
print(f'Accuracy scores for each fold: {scores}')
print(f'Average accuracy: {scores.mean()}')
