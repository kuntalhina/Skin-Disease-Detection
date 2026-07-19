import os

print("Training Data Distribution:\n")

for cls in os.listdir("dataset/train"):
    print(cls, ":", len(os.listdir(f"dataset/train/{cls}")))