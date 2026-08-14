import os

# Path to the folder containing your images
folder_path = 'E:/SIH/dataset/train_1/s2'

# Change directory to the folder
os.chdir(folder_path)

# Get a list of files in the directory
files = os.listdir(folder_path)

# Sort the files to ensure they are renamed in a specific order
files.sort()

# Rename each file
for index, file in enumerate(files):
    # Create a new name for the file
    new_name = f'opt_img{index + 1}{os.path.splitext(file)[1]}'
    # Rename the file
    os.rename(file, new_name)

print("Renaming completed.")
