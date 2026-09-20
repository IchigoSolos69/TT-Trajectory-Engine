import os
import shutil
import random

def create_val_split():
    train_img_dir = "dataset/images/train"
    train_lbl_dir = "dataset/labels/train"
    val_img_dir = "dataset/images/val"
    val_lbl_dir = "dataset/labels/val"

    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    if not os.path.exists(train_img_dir):
        print(f"Error: {train_img_dir} does not exist.")
        return

    images = [f for f in os.listdir(train_img_dir) if f.endswith('.jpg')]
    if not images:
        print("No images found in training set.")
        return

    # Random 10% validation split
    random.seed(42)
    val_sample_size = int(len(images) * 0.1)
    val_images = random.sample(images, val_sample_size)

    print(f"Splitting {len(val_images)} images into validation set...")

    for img_file in val_images:
        lbl_file = img_file.replace('.jpg', '.txt')
        
        src_img = os.path.join(train_img_dir, img_file)
        dst_img = os.path.join(val_img_dir, img_file)
        shutil.move(src_img, dst_img)

        src_lbl = os.path.join(train_lbl_dir, lbl_file)
        dst_lbl = os.path.join(val_lbl_dir, lbl_file)
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)

    print(f"Validation split complete! Train: {len(os.listdir(train_img_dir))}, Val: {len(os.listdir(val_img_dir))}")

if __name__ == "__main__":
    create_val_split()
