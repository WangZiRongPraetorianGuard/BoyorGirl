import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier

## It's a failure method because girl's sample are relatively smaller than boy
 
# 读取CSV文件
data = pd.read_csv(r"C:\Users\Hank\BoyorGirl\BoyorGirl\KNN\outlier_detect\KNN_after_include_bmi.csv")

# 拆分男生和女生数据
male_data = data[data['gender'] == 1]
female_data = data[data['gender'] == 2]

# 提取特征和标签
columns_to_impute = ['star_sign', 'phone_os', 'height', 'weight', 'sleepiness', 'iq', 'fb_friends', 'yt','bmi']
X_male = male_data[columns_to_impute]
y_male = male_data['gender']
X_female = female_data[columns_to_impute]
y_female = female_data['gender']

try:
    # 初始化StratifiedShuffleSplit对象
    stratified_splitter_male = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    stratified_splitter_female = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # 使用split方法拆分数据集，此处只进行一次拆分
    for train_index, test_index in stratified_splitter_male.split(X_male, y_male):
        X_male_train, X_male_test = X_male.iloc[train_index], X_male.iloc[test_index]
        y_male_train, y_male_test = y_male.iloc[train_index], y_male.iloc[test_index]

    for train_index, test_index in stratified_splitter_female.split(X_female, y_female):
        X_female_train, X_female_test = X_female.iloc[train_index], X_female.iloc[test_index]
        y_female_train, y_female_test = y_female.iloc[train_index], y_female.iloc[test_index]

    # 使用随机森林模型
    rf_male = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_female = RandomForestClassifier(n_estimators=100, random_state=42)

    # 训练男生模型
    rf_male.fit(X_male_train, y_male_train)

    # 训练女生模型
    rf_female.fit(X_female_train, y_female_train)

    # 评估男生模型
    male_score = rf_male.score(X_male_test, y_male_test)
    print("Male Model Accuracy:", male_score)

    # 评估女生模型
    female_score = rf_female.score(X_female_test, y_female_test)
    print("Female Model Accuracy:", female_score)

except ValueError:
    print("One of the genders has no samples, unable to perform stratified sampling.")

