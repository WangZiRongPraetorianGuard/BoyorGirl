import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.pipeline import Pipeline
from skrebate import ReliefF

# 假设df已经按照您的代码加载和预处理
df = pd.read_csv(r'C:\AI專案\InformationData\Boys and girls\BoyorGirl\HotDeck\data\AfterHotDeck\KNN_withBMI.csv')

# 转换性别标签
df['gender'] = df['gender'].apply(lambda x: 0 if x == 1 else 1)


# 定义特征和目标变量
X = df.drop(columns=['gender'])
y = df['gender'].values

best_score = 0
best_k = 0

# ReliefF不直接支持选择k个最佳特征，所以我们先单独运行ReliefF来评估特征的重要性
relief = ReliefF(n_neighbors=100) # n_neighbors是ReliefF的一个参数，可根据需要调整
relief.fit(X.values, y)


# 根据ReliefF得分对特征进行排序
features_ranked_by_importance = relief.feature_importances_.argsort()[::-1]

# 遍历1到9的特征数量
for k in range(1, 10):
    selected_features_indices = features_ranked_by_importance[:k]
    X_selected = X.iloc[:, selected_features_indices]

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
    ])
    
    # 应用交叉验证并计算平均准确率
    scores = cross_val_score(pipeline, X_selected, y, cv=5)
    mean_score = scores.mean()
    
    # 检查是否为最佳得分
    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print(f'Best number of features: {best_k}, Best Average Accuracy: {best_score}')

# 使用最佳k值选择特征并再次拟合模型
selected_features_indices_final = features_ranked_by_importance[:best_k]
X_selected_final = X.iloc[:, selected_features_indices_final]
pipeline.fit(X_selected_final, y)

print(f'Selected features for k={best_k}: {X.columns[selected_features_indices_final]}')
