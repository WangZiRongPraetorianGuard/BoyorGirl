import pandas as pd

def calculate_bmi(height, weight):
    """
    计算BMI
    :param height: 身高（单位：米）
    :param weight: 体重（单位：公斤）
    :return: BMI值
    """
    if height == 0:
        return None

    return weight / ((height/100) ** 2)

df = pd.read_csv(r'C:\Users\Hank\BoyorGirl\BoyorGirl\KNN\dataset\ExtraTreesRegressor_mofified.csv')

# 计算BMI并添加到DataFrame中
df['bmi'] = df.apply(lambda row: calculate_bmi(row['height'], row['weight']), axis=1)

df.to_csv(r'C:\Users\Hank\BoyorGirl\BoyorGirl\KNN\dataset\ExtraTreesRegressor_modified_with_bmi.csv', index=False)
