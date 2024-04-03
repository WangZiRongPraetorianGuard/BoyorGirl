import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import lightgbm as lgb
from sklearn.metrics import accuracy_score

# 加载数据
df = pd.read_csv(r'C:\AI專案\InformationData\Boys and girls\BoyorGirl\HotDeck\data\AfterHotDeck\Train\Train_Medium_WithBMI.csv')

# 准备特征和目标变量
X = df[['star_sign', 'phone_os', 'height', 'weight', 'sleepiness', 'iq', 'fb_friends', 'yt', 'bmi']]
y = df['gender'] - 1  # 转换类别标签从[1 2]为[0 1]

# 准备交叉验证
cv = StratifiedKFold(n_splits=5)
scores = []

# 遍历交叉验证的每个分割
for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 创建并应用管道
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif, k='all')),  # 可以调整k为所需的特征数
        ('lgb', lgb.LGBMClassifier(objective='binary'))
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    score = accuracy_score(y_test, predictions)
    scores.append(score)
    
    # 获取并打印选中的特征
    selected_features = X_train.columns[pipeline.named_steps['feature_selection'].get_support()]
    print(f"Selected features in this fold: {selected_features.tolist()}")

# 打印准确率
print(f'每次交叉验证的准确率: {scores}')
print(f'平均准确率: {sum(scores) / len(scores)}')
