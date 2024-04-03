import pandas as pd
from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler 

# 讀取訓練資料集
train_data = pd.read_csv(r"C:\Users\Hank\BoyorGirl\BoyorGirl\KNN\outlier_detect\dataset_without_outlier_by_KNN.csv")

# 將性別標籤設置為0和1，其中1代表男性，2代表女性
train_data['gender'] = train_data['gender'].apply(lambda x: 1 if x == 1 else 0)

# 將資料集分為特徵（X）和標籤（y）
X = train_data.drop(columns=['gender'])
y = train_data['gender']

# 初始化標準化器
scaler = StandardScaler()

# 對資料進行標準化
X_scaled = scaler.fit_transform(X)

# 初始化 CatBoost 分類器
catboost = CatBoostClassifier(random_state=42, verbose=0)

# 使用交叉验证评估模型性能
cv_scores = cross_val_score(catboost, X_scaled, y, cv=5, scoring='accuracy')

print("Cross Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())
