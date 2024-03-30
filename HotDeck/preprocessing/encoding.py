
import pandas as pd

# Load your data
# Replace 'path_to_your_file.csv' with the actual path to your CSV file
df = pd.read_csv(r'C:\AI專案\InformationData\Boys and girls\BoyorGirl\HotDeck\data\boy or girl 2024 test no ans_missingValue.csv')

# Define a mapping for star_signs
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

# Apply the mapping to the star_sign column
df['star_sign'] = df['star_sign'].map(star_sign_mapping)

# Define a mapping for phone_os
phone_os_mapping = {
    'Apple': 0,
    'Android': 1
}

# Apply the mapping to the phone_os column
df['phone_os'] = df['phone_os'].map(phone_os_mapping)

# If there are any NaNs from mapping, you may want to fill them with a default value or drop the rows
df['phone_os'] = df['phone_os'].fillna(0)  # Assuming 'Apple' as default
df = df.drop(["self_intro"], axis=1)
# Save the DataFrame back to a CSV if needed
df.to_csv(r'C:\AI專案\InformationData\Boys and girls\BoyorGirl\HotDeck\data\EncodingTest.csv', index=False)

# Print the DataFrame to verify changes
print(df)
