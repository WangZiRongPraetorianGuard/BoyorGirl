import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler 

# 讀取訓練資料集
train_data = pd.read_csv(r"C:\Users\Hank\BoyorGirl\BoyorGirl\KNN\dataset\ExtraTreesRegressor_modified_with_bmi.csv")

# 讀取測試資料集
test_data = pd.read_csv(r"C:\Users\Hank\BoyorGirl\BoyorGirl\KNN\dataset\test_KNN_with_bmi.csv")

test_data = test_data.drop(columns=['self_intro'])

# 將性別標籤設置為0和1，其中1代表男性，0代表女性
train_data['gender'] = train_data['gender'].apply(lambda x: 1 if x == 1 else 0)
test_data['gender'] = test_data['gender'].apply(lambda x: 1 if x == 1 else 0)

# 將資料集分為特徵（X）和標籤（y）
X = train_data.drop(columns=['gender'])
y = train_data['gender']
X_test = test_data.drop(columns=['gender'])

# 初始化標準化器
scaler = StandardScaler()

# 對資料進行標準化
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 初始化CatBoost分類器
catboost = CatBoostClassifier(random_state=42)

# 定義超參數範圍
param_grid = {
    'iterations': [100, 200, 300],  # 迭代次数
    'learning_rate': [0.01, 0.05, 0.1],  # 学习率
    'depth': [4, 6, 8],  # 树的深度
}

# 使用 RapidSearch (GridSearchCV) 進行超參數優化
grid_search = GridSearchCV(estimator=catboost, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_scaled, y)

# 獲取最佳模型
best_catboost = grid_search.best_estimator_

# 在測試資料集上進行預測
test_predictions = best_catboost.predict(X_test_scaled)

# 建立新的 DataFrame 來存放預測結果
result_df = pd.DataFrame({'ID': range(1, len(test_predictions) + 1), 'gender': [2 if pred == 0 else pred for pred in test_predictions]})

# 將結果存入新的 CSV 檔案中
result_df.to_csv(r'C:\Users\Hank\BoyorGirl\BoyorGirl\KNN\prediction_result\cat_extra_KNN_prediction_result.csv', index=False)

# 使用交叉验证评估模型性能
cv_scores = cross_val_score(best_catboost, X_scaled, y, cv=5, scoring='accuracy')

print("Cross Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())
