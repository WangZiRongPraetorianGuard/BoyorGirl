import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# 加载数据
df = pd.read_csv(r'C:\Users\Hank\BoyorGirl\BoyorGirl\HotDeck\data\AfterHotDeck\TrainHotDeck_withBMI.csv')

# 分割数据集（不进行实际的数据划分，只定义特征和目标）
X = df.drop(columns=['gender'])
y = df['gender']

best_score = 0
best_k = 0

# 遍历1到9的特征数量
for k in range(1, 10):  # 包括9个特征
    # 创建包括预处理步骤和 KNN 模型的管道
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif, k=k)),
        ('knn', KNeighborsClassifier())
    ])
    
    # 应用交叉验证并计算平均准确率
    scores = cross_val_score(pipeline, X, y, cv=5)
    mean_score = scores.mean()
    
    # 检查是否为最佳得分
    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print(f'Best number of features: {best_k}, Best Average Accuracy: {best_score}')

# 确定最佳特征
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif, k=best_k)),
    ('knn', KNeighborsClassifier())
])

pipeline.fit(X, y)
selected_features = pipeline.named_steps['feature_selection'].get_support(indices=True)
print(f'Selected features for k={best_k}: {X.columns[selected_features]}')
