import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 加载数据
df = pd.read_csv(r'C:\AI專案\InformationData\Boys and girls\BoyorGirl\HotDeck\data\AfterHotDeck\TrainHotDeck_withBMI.csv')

# 数据预处理
# 处理缺失值
imputer = SimpleImputer(strategy='mean')
df[['height', 'weight', 'sleepiness', 'iq', 'fb_friends', 'yt', 'bmi']] = imputer.fit_transform(df[['height', 'weight', 'sleepiness', 'iq', 'fb_friends', 'yt', 'bmi']])

# 定义特征和目标变量
X = df[['star_sign', 'phone_os', 'height', 'weight', 'sleepiness', 'iq', 'fb_friends', 'yt', 'bmi']]
y = df['gender']

# 创建一个带数据预处理的SVM模型管道
clf_pipeline = make_pipeline(StandardScaler(), SVC(kernel='linear'))

# 应用交叉验证
# cv 参数表示交叉验证分割数目
scores = cross_val_score(clf_pipeline, X, y, cv=5)

# 打印每次交叉验证的准确率以及平均准确率
print(f'Accuracy scores for each fold: {scores}')
print(f'Average accuracy: {scores.mean()}')
