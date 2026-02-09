import os
import shutil
import re
import argparse

# Set up argument parsing to accept root_path as input
parser = argparse.ArgumentParser(description="Organize images in directories.")
parser.add_argument('root_path', type=str, help="Path to the root directory")
args = parser.parse_args()

root_path = args.root_path

directories = []

order = ['form.jpg', 'id.jpg', 'exp.jpg']

for dir_path, dirnames, files in os.walk(root_path):
    # Add the directory to the list if it contains files
    if files:
        print("Directory: ", dir_path)
        directories.append(dir_path)

# Now loop through the collected directories
for dir_path in directories:
    numbers = {}
    # Loop through the files in each directory
    for file in os.listdir(dir_path):
        # Process only .jpg files
        if file.endswith('.jpg') or file.endswith('.png'):
            fulpath = os.path.join(dir_path, file)

            # Rename 'Image.jpg' to 'form.jpg'
            if file == 'Image.jpg':
                numbers[0] = file
            
            # If the filename contains a number in parentheses
            else:
                match = re.search(r'\d+', file)
                if match:
                    # Extract the number found inside parentheses
                    number = int(match.group(0))
                    numbers[number] = file

        # Keep 'form.jpg' as the new name
        new_name = os.path.join(dir_path, 'form.jpg')

    sorted_dict = {key: numbers[key] for key in sorted(numbers)}

    try:
        for i, path in enumerate(sorted_dict.values()):
            shutil.move(os.path.join(dir_path, path), os.path.join(dir_path, order[i]))
    except Exception as e:
        print('Maybe you have more than 3 images, {e}, Continuing the renaming')

