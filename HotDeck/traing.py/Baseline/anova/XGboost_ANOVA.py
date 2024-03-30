import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
from sklearn.pipeline import Pipeline

# 加载数据
df = pd.read_csv(r'C:\AI專案\InformationData\Boys and girls\BoyorGirl\HotDeck\data\AfterHotDeck\KNN_withBMI.csv')

# 转换性别标签
df['gender'] = df['gender'].apply(lambda x: 0 if x == 1 else 1)

# 定义特征和目标变量
X = df.drop(columns=['gender'])
y = df['gender'].values

best_score = 0
best_k = 0

# 遍历1到9的特征数量
for k in range(1, 10):  # 包括9个特征
    # 创建包括预处理步骤、ANOVA F-test 特征选择和 XGBoost 模型的管道
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif, k=k)),
        ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
    ])
    
    # 应用交叉验证并计算平均准确率
    scores = cross_val_score(pipeline, X, y, cv=5)
    mean_score = scores.mean()
    
    # 检查是否为最佳得分
    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print(f'Best number of features: {best_k}, Best Average Accuracy: {best_score}')

# 使用最佳 k 值创建并拟合管道，以查看选择了哪些特征
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif, k=best_k)),
    ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
])
pipeline.fit(X, y)

selected_features = pipeline.named_steps['feature_selection'].get_support(indices=True)
print(f'Selected features for k={best_k}: {X.columns[selected_features]}')
