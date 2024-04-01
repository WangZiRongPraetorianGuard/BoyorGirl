import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler 

# 讀取訓練資料集
train_data = pd.read_csv(r"C:\Users\Hank\BoyorGirl\BoyorGirl\KNN\outlier_detect\dataset_without_outlier_by_KNN.csv")

# 將self_intro欄位從訓練資料中移除，因為這裡不打算使用該欄位作為特徵
# train_data = train_data.drop(columns=['self_intro'])

# 將性別標籤設置為0和1，其中1代表男性，2代表女性
train_data['gender'] = train_data['gender'].apply(lambda x: 1 if x == 1 else 0)

# 將資料集分為特徵（X）和標籤（y）
X = train_data.drop(columns=['gender'])
y = train_data['gender']

# 將資料集分為訓練集和驗證集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化標準化器
scaler = StandardScaler()

# 對訓練集和驗證集進行標準化
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 初始化隨機森林分類器
rf = RandomForestClassifier(random_state=42)

# 定義超參數範圍
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 使用 RapidSearch (GridSearchCV) 進行超參數優化
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# 獲取最佳模型
best_rf = grid_search.best_estimator_

# 在驗證集上進行預測
val_predictions = best_rf.predict(X_val_scaled)

# 計算模型在驗證集上的準確率
accuracy = accuracy_score(y_val, val_predictions)
print("Validation Accuracy:", accuracy)

# 進行測試資料集的預測
test_data = pd.read_csv(r"C:\Users\Hank\BoyorGirl\BoyorGirl\KNN\dataset\test_KNN_without_outlier.csv")

# 將測試資料集進行與訓練資料相同的預處理
test_data['gender'] = test_data['gender'].apply(lambda x: 1 if x == 1 else 0)
test_data = test_data.drop(columns=['self_intro'])
test_data_scaled = scaler.transform(test_data)
test_predictions = best_rf.predict(test_data_scaled)

# 建立新的 DataFrame 來存放預測結果
result_df = pd.DataFrame({'ID': range(1, len(test_predictions) + 1), 'gender': [2 if pred == 0 else pred for pred in test_predictions]})

# 將結果存入新的 CSV 檔案中
result_df.to_csv('complete_prediction_result_RF_rapid.csv', index=False)

# 輸出預測結果
print(result_df)
