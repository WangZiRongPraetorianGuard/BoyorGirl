from tpot import TPOTClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

# 加载和预处理数据
df = pd.read_csv(r'C:\AI專案\InformationData\Boys and girls\BoyorGirl\HotDeck\data\AfterHotDeck\KNN_withBMI.csv')
df['gender'] = df['gender'].apply(lambda x: 0 if x == 1 else 1)

X = df.drop(columns=['gender'])
y = df['gender'].values

# 创建TPOT分类器
tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42, scoring='accuracy', cv=5)
# 训练TPOT分类器
tpot.fit(X, y)

# 可以导出最佳管道到Python脚本
tpot.export('tpot_best_pipeline.py')
