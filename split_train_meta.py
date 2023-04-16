
import os
import pandas as pd
from time import time, sleep

base_path = "/home/zhangbw/Documents/projects/iceCube-kaggle/kaggle/icecube-neutrinos-in-deep-ice/"
meta_path = os.path.join(base_path, "train_meta.parquet")
splitted_path = os.path.join(base_path, "train_meta")

if not os.path.exists(splitted_path):
    os.mkdir(splitted_path)

start = time()
meta_train = pd.read_parquet(meta_path)
print(f"read meta spent {time() - start} s")

sleep(5)
print(f"wake up")
# meta_train = meta_train[meta_train["batch_id"] < 300]
meta_train = meta_train[meta_train["batch_id"] <= 10]

sleep(5)
print(f"wake up")

for i, df in meta_train.groupby("batch_id"):
    print(f"processing {i} -> {df.shape}")
    splitted = os.path.join(splitted_path, f"meta_{i}.parquet")
    df.to_parquet(splitted)

print(df)
