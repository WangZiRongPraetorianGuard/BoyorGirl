import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# 加载数据
df = pd.read_csv(r'C:\AI專案\InformationData\Boys and girls\BoyorGirl\HotDeck\data\AfterHotDeck\TrainHotDeck_withBMI.csv')

# 数据预处理
# 处理缺失值
imputer = SimpleImputer(strategy='mean')
df_imputed = imputer.fit_transform(df.drop(columns=['gender']))
df_scaled = StandardScaler().fit_transform(df_imputed)  # 标准化
X = df_scaled
y = df['gender'].values

# 特征数量范围
features_range = range(1, 10)  # 假设你有9个特征

best_score = 0
best_k = 0

# 遍历不同的特征数量 k
for k in features_range:
    # 创建管道，包括 ANOVA F-test 特征选择和 SVM 模型
    clf_pipeline = Pipeline([
        ('anova', SelectKBest(f_classif, k=k)),
        ('svm', SVC(kernel='linear'))
    ])
    
    # 应用交叉验证并计算平均准确率
    scores = cross_val_score(clf_pipeline, X, y, cv=5)
    mean_score = scores.mean()
    print(f'Number of features: {k}, Cross-validation Accuracy: {mean_score}')
    
    # 更新最佳分数和最佳特征数量
    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print(f'Best number of features: {best_k}, Best cross-validation accuracy: {best_score}')
