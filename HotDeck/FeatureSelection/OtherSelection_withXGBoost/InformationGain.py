import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline

# 加载和预处理数据
df = pd.read_csv(r'C:\AI專案\InformationData\Boys and girls\BoyorGirl\HotDeck\processed_dataset.csv')
df['gender'] = df['gender'].apply(lambda x: 0 if x == 1 else 1)

X = df.drop(columns=['gender'])
y = df['gender'].values

best_score = 0
best_k = 0

# 训练决策树来获取特征重要性
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X, y)
feature_importances = tree.feature_importances_

# 根据特征重要性对特征进行排序
features_ranked_by_importance = feature_importances.argsort()[::-1]

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
