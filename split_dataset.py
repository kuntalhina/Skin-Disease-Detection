import os
import random
import shutil

base_path = "dataset"
train_path = os.path.join(base_path, "train")
test_path = os.path.join(base_path, "test")

classes = ["melanoma", "nevus", "bcc"]

split_ratio = 0.2  # 20% test data

for cls in classes:
    class_train_path = os.path.join(train_path, cls)
    class_test_path = os.path.join(test_path, cls)

    os.makedirs(class_test_path, exist_ok=True)

    images = os.listdir(class_train_path)
    random.shuffle(images)

    split_size = int(len(images) * split_ratio)
    test_images = images[:split_size]

    for img in test_images:
        src = os.path.join(class_train_path, img)
        dst = os.path.join(class_test_path, img)
        shutil.move(src, dst)

print("Train-Test Split Completed!")