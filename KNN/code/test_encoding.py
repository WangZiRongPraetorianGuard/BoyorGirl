import pandas as pd

# 讀取測試資料集
test_data = pd.read_csv(r"C:\Users\Hank\BoyorGirl\BoyorGirl\boy or girl 2024 test no ans_missingValue.csv")

# 星座的映射字典
star_sign_mapping = {
    '水瓶座': 0,
    '雙魚座': 1,
    '牡羊座': 2,
    '金牛座': 3,
    '雙子座': 4,
    '巨蟹座': 5,
    '獅子座': 6,
    '處女座': 7,
    '天秤座': 8,
    '天蠍座': 9,
    '射手座': 10,
    '摩羯座': 11
}

# 將星座欄位轉換為數字
test_data['star_sign'] = test_data['star_sign'].map(star_sign_mapping)

# 手機操作系統的映射字典
phone_os_mapping = {
    'Apple': 0,
    'Android': 1,
    'Windows phone': 2  
}

# 將手機操作系統欄位轉換為數字
test_data['phone_os'] = test_data['phone_os'].map(phone_os_mapping)

# 輸出處理後的測試資料集
test_data.to_csv(r"C:\Users\Hank\BoyorGirl\BoyorGirl\test_encoding.csv", index=False)