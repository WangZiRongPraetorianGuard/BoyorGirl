import pandas as pd

# 讀取資料
data = pd.read_csv(r"C:\Users\Hank\BoyorGirl\BoyorGirl\test_KNN.csv")

# 定義合理的身高範圍
min_height = 150  # 最小合理身高（厘米）
max_height = 200  # 最大合理身高（厘米）

# 定義合理的體重範圍
min_weight = 40  # 最小合理體重（公斤）
max_weight = 120  # 最大合理體重（公斤）

# 使用盒形圖找出離群值的閾值
Q1 = data['fb_friends'].quantile(0.25)
Q3 = data['fb_friends'].quantile(0.75)
IQR = Q3 - Q1
fb_lower_bound = Q1 - 1.5 * IQR
fb_upper_bound = Q3 + 1.5 * IQR

# 使用盒形圖找出離群值的閾值
Q1 = data['yt'].quantile(0.25)
Q3 = data['yt'].quantile(0.75)
IQR = Q3 - Q1
yt_lower_bound = Q1 - 1.5 * IQR
yt_upper_bound = Q3 + 1.5 * IQR

# 將離群值截斷為最接近的合理值
data['height'] = data['height'].clip(lower=min_height, upper=max_height)
data['weight'] = data['weight'].clip(lower=min_weight, upper=max_weight)
data['fb_friends'] = data['fb_friends'].clip(lower=fb_lower_bound, upper=fb_upper_bound)
data['yt'] = data['yt'].clip(lower=yt_lower_bound, upper=yt_upper_bound)

data.to_csv(r"C:\Users\Hank\BoyorGirl\BoyorGirl\test_KNN_without_outlier.csv", index=False)