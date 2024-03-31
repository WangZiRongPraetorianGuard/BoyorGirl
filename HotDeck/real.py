import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

# 加载数据集
train_path = r'C:\AI專案\InformationData\Boys and girls\BoyorGirl\HotDeck\data\AfterHotDeck\Train\dataset_without_outlier_by_KNNBNI.csv'
test_path = r'C:\AI專案\InformationData\Boys and girls\BoyorGirl\HotDeck\data\AfterHotDeck\Test\test_KNN_without_outlierBNI.csv'
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# 准备训练集的特征和目标变量
X_train = df_train[['star_sign', 'phone_os', 'height', 'weight', 'fb_friends', 'sleepiness', 'iq', 'yt', 'bmi']]
y_train = df_train['gender'] - 1  # 确保类别标签从0开始

# 准备测试集的特征（不包含目标变量）
X_test = df_test[['star_sign', 'phone_os', 'height', 'weight', 'fb_friends', 'sleepiness', 'iq', 'yt', 'bmi']]

# 创建一个包含预处理步骤和CatBoost模型的管道
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # 缺失值处理
    ('scaler', StandardScaler()),  # 特征缩放
    ('catboost', CatBoostClassifier(verbose=0, auto_class_weights='Balanced'))  # CatBoost模型
])

# 训练模型
pipeline.fit(X_train, y_train)

# 使用模型进行预测，并将结果从0和1调整为1和2
predictions = pipeline.predict(X_test) + 1

# 将ID和预测的性别保存为CSV文件，这里性别为1和2
output = pd.DataFrame({'id': df_test['id'], 'predicted_gender': predictions})
output.to_csv('predicted_genders.csv', index=False)

print("预测完成，结果已保存为 predicted_genders.csv")
