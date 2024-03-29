import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler 

# 讀取訓練資料集
train_data = pd.read_csv(r"C:\Users\Hank\BoyorGirl\BoyorGirl\KNN_without_outlier.csv")

# 將self_intro欄位從訓練資料中移除，因為這裡不打算使用該欄位作為特徵
train_data = train_data.drop(columns=['self_intro'])

# 將性別標籤設置為0和1，其中1代表男性，2代表女性
train_data['gender'] = train_data['gender'].apply(lambda x: 1 if x == 1 else 0)

# 將資料集分為特徵（X）和標籤（y）
X = train_data.drop(columns=['gender'])
y = train_data['gender']

# 將資料集分為訓練集和驗證集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化並訓練隨機森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 初始化標準化器
scaler = StandardScaler()

# 對訓練集進行標準化
X_train_scaled = scaler.fit_transform(X_train)

# 在訓練集上訓練模型
model.fit(X_train_scaled, y_train)

# 在驗證集上進行預測
X_val_scaled = scaler.transform(X_val)
val_predictions = model.predict(X_val_scaled)

# 計算模型在驗證集上的準確率
accuracy = accuracy_score(y_val, val_predictions)
print("Validation Accuracy:", accuracy)

# 讀取測試資料集
test_data = pd.read_csv(r"C:\Users\Hank\BoyorGirl\BoyorGirl\test_KNN_without_outlier.csv")

# 將self_intro欄位從測試資料中移除
test_data = test_data.drop(columns=['self_intro', 'id', 'gender'])

# 對測試資料進行標準化
test_data_scaled = scaler.transform(test_data)

# 進行預測
test_predictions = model.predict(test_data_scaled)

# 將預測結果轉換為原始的1和2標籤
# test_predictions = ['男生' if pred == 1 else '女生' for pred in test_predictions]

# 建立新的 DataFrame 來存放預測結果
result_df = pd.DataFrame({'ID': range(1, len(test_predictions) + 1), 'gender': [2 if pred == 0 else pred for pred in test_predictions]})

# 將結果存入新的 CSV 檔案中
result_df.to_csv('prediction_result.csv', index=False)

# 輸出預測結果
print(result_df)

