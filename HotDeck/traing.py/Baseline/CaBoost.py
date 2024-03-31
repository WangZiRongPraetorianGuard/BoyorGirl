import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

# 加载数据
df = pd.read_csv(r'C:\AI專案\InformationData\Boys and girls\BoyorGirl\HotDeck\data\AfterHotDeck\Train\Train_Medium_WithBMI.csv')

# 准备特征和目标变量
# X = df[['star_sign', 'phone_os', 'height', 'weight', 'sleepiness', 'iq', 'fb_friends', 'yt', 'bmi']]
X = df[[ 'phone_os', 'height', 'weight', 'fb_friends','sleepiness', 'iq', 'yt', 'bmi']]
y = df['gender'] - 1  # 确保类别标签从0开始

# 创建一个包含预处理步骤和CatBoost模型的管道
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # 缺失值处理
    ('scaler', StandardScaler()),  # 特征缩放
    ('catboost', CatBoostClassifier(verbose=0, auto_class_weights='Balanced'))  # CatBoost模型
])

# 应用交叉验证
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')  # cv参数表示交叉验证分割数目

# 打印每次交叉验证的准确率以及平均准确率
print('每次交叉验证的准确率:', scores)
print('平均准确率:', scores.mean())
