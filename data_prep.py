import os
import shutil

def copy_every_other_image(source_folder, target_folder):
    # Ensure target folder exists
    os.makedirs(target_folder, exist_ok=True)

    # List all files in the source folder
    files = sorted(os.listdir(source_folder))
    
    # Copy every other file (1st, 3rd, 5th, etc.)
    for i, file in enumerate(files):
        if i % 4 == 0:  # This will pick every other file
            src_path = os.path.join(source_folder, file)
            dest_path = os.path.join(target_folder, file)
            shutil.copy(src_path, dest_path)

    print(f"Copied {len(files)//4} images to {target_folder}")

# Example usage
source_folder = 'D:/v_2/urban/s1'
target_folder = 'dataset/train_half/s1'
copy_every_other_image(source_folder, target_folder)
