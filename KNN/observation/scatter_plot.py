import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 讀取資料集
data = pd.read_csv(r"C:\Users\Hank\BoyorGirl\BoyorGirl\KNN\dataset\KNN_without_outlier.csv")

# 保留資料集中的數值型特徵（假設資料集中的特徵都是數值型）
numeric_data = data.select_dtypes(include=['int', 'float'])

# 繪製 pairplot
# sns.pairplot(numeric_data)
# plt.show()

# 計算特徵之間的相關係數
correlation_matrix = numeric_data.corr()

# 繪製熱力圖
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()