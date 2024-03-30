import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.pipeline import Pipeline, make_pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# 加载和预处理数据
df = pd.read_csv(r'C:\AI專案\InformationData\Boys and girls\BoyorGirl\HotDeck\data\AfterHotDeck\KNN_withBMI.csv')
df['gender'] = df['gender'].apply(lambda x: 0 if x == 1 else 1)

X = df.drop(columns=['gender'])
y = df['gender'].values

# 创建一个XGBoost分类器实例
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# 使用Sequential Forward Selection
sfs = SFS(xgb_clf, 
          k_features=9,  # 最多选择特征的数量，可以根据需要调整
          forward=True, 
          floating=False, 
          scoring='accuracy',
          cv=5)

# 创建一个包含预处理步骤的管道
pipeline = make_pipeline(SimpleImputer(strategy='mean'),
                         StandardScaler(),
                         sfs)

# 拟合管道
pipeline.fit(X, y)

# 打印选择的特征
selected_features = list(sfs.k_feature_idx_)
print(f'Selected features indices: {selected_features}')
print(f'Selected features names: {X.columns[selected_features]}')

# 评估模型性能
scores = cross_val_score(xgb_clf, X.iloc[:, selected_features], y, cv=5)
print(f'Average Accuracy: {scores.mean()}')
