import numpy as np
import pickle
import os

print("Creating dummy dataset...")
os.makedirs("dummy_dataset", exist_ok=True)

# vocab_size = 10
# block_size for dry run will be 64, make data larger
train_data_list = []
for _ in range(20): # make it long enough for a few blocks
    train_data_list.extend([i % 10 for i in range(10)])
train_data = np.array(train_data_list, dtype=np.uint16)

val_data_list = []
for _ in range(20):
    val_data_list.extend([(i + 5) % 10 for i in range(10)])
val_data = np.array(val_data_list, dtype=np.uint16)

train_data.tofile("dummy_dataset/train.bin")
val_data.tofile("dummy_dataset/val.bin")

meta = {'vocab_size': 10, 'itos': {i: str(i) for i in range(10)}, 'stoi': {str(i): i for i in range(10)}}
with open("dummy_dataset/meta.pkl", "wb") as f:
    pickle.dump(meta, f)

print("Dummy dataset created in 'dummy_dataset' directory.")
print("Contents of dummy_dataset:")
for item in os.listdir("dummy_dataset"):
    print(item)
