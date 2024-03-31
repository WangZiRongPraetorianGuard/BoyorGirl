import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# 加载数据
df = pd.read_csv(r'C:\AI專案\InformationData\Boys and girls\BoyorGirl\HotDeck\data\AfterHotDeck\Train\Train_Medium_WithBMI.csv')

# 准备特征和目标变量
X = df[['star_sign', 'phone_os', 'height', 'weight', 'sleepiness', 'iq', 'fb_friends', 'yt', 'bmi']]
y = df['gender'] - 1  # 确保类别标签从0开始

# 定义特征名称列表，以便后续打印
feature_names = X.columns

# 创建交叉验证器
cv = StratifiedKFold(n_splits=5)

# 创建一个包含预处理步骤和CatBoost模型的管道
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # 缺失值处理
    ('scaler', StandardScaler()),  # 特征缩放
    ('feature_selection', SelectKBest(f_classif, k='all')),  # 特征选择
    ('catboost', CatBoostClassifier(verbose=0, auto_class_weights='Balanced'))  # CatBoost模型
])

# 用于存储每次交叉验证的准确率
scores = []

for train_index, test_index in cv.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # 拟合管道
    pipeline.fit(X_train, y_train)
    
    # 预测并计算准确率
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    scores.append(accuracy)
    
    # 获取选中的特征
    support = pipeline.named_steps['feature_selection'].get_support()
    selected_features = feature_names[support]
    
    print("Selected features:", selected_features)

# 打印每次交叉验证的准确率以及平均准确率
print('每次交叉验证的准确率:', scores)
print('平均准确率:', sum(scores) / len(scores))
