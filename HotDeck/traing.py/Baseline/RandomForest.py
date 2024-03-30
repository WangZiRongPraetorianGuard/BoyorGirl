import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 加载数据
df = pd.read_csv(r'C:\AI專案\InformationData\Boys and girls\BoyorGirl\HotDeck\data\AfterHotDeck\KNN_withoutbmi.csv')

# 数据预处理
# 处理缺失值
imputer = SimpleImputer(strategy='mean')
df_imputed = imputer.fit_transform(df.drop(columns=['gender']))
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_imputed)  # 标准化

# 定义特征和目标变量
X = df_scaled
y = df['gender'].values

# 创建 KNN 模型
knn = KNeighborsClassifier()  # 使用默认参数

# 创建包括预处理步骤的管道
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # 处理缺失值
    ('scaler', StandardScaler()),  # 数据标准化
    ('knn', KNeighborsClassifier())  # KNN 模型
])

# 应用交叉验证
scores = cross_val_score(pipeline, X, y, cv=5)  # cv 参数表示交叉验证分割数目

# 打印每次交叉验证的准确率以及平均准确率
print(f'Accuracy scores for each fold: {scores}')
print(f'Average accuracy: {scores.mean()}')
