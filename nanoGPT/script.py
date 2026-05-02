import os
import glob
import pickle
import numpy as np

src_dir = "../build-nanogpt/edu_fineweb10B"
dst_dir = "data/fineweb"
os.makedirs(dst_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(src_dir, "*.npy")))
if not files:
    raise RuntimeError(f"No .npy files found in {src_dir}")

total_tokens = 0
for f in files:
    arr = np.load(f, mmap_mode="r")
    total_tokens += len(arr)

split = int(0.99 * total_tokens)

train_path = os.path.join(dst_dir, "train.bin")
val_path = os.path.join(dst_dir, "val.bin")

written = 0

with open(train_path, "wb") as f_train, open(val_path, "wb") as f_val:
    for f in files:
        arr = np.load(f, mmap_mode="r")
        n = len(arr)

        if written >= split:
            arr.astype(np.uint16).tofile(f_val)
        elif written + n <= split:
            arr.astype(np.uint16).tofile(f_train)
        else:
            cut = split - written
            arr[:cut].astype(np.uint16).tofile(f_train)
            arr[cut:].astype(np.uint16).tofile(f_val)

        written += n
        print(f"processed {os.path.basename(f)} ({written}/{total_tokens} tokens)")

meta = {"vocab_size": 50257}
with open(os.path.join(dst_dir, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("done")
print(f"train.bin: {train_path}")
print(f"val.bin:   {val_path}")
