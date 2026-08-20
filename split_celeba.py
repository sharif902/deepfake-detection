import os
import shutil
import pandas as pd

# Read partition file
df_part = pd.read_csv('dataset/img_align_celeba/list_eval_partition.csv')

# 2 = test split
test_df = df_part[df_part['partition'] == 2]

print(f"Total test images: {len(test_df)}")

# Create test folder
os.makedirs('test/celeba_test', exist_ok=True)

# Copy test images to test folder
for idx, row in test_df.iterrows():
    src = os.path.join('dataset/img_align_celeba/img_align_celeba', row['image_id'])
    dst = os.path.join('test/celeba_test', row['image_id'])
    if os.path.exists(src):
        shutil.copy(src, dst)

print("Done! CelebA test images copied to test/celeba_test!")