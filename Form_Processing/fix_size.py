import glob
import os
from PIL import Image
from tqdm import tqdm

# Define the file patterns to search for
patterns = ["**/form_c.jpg", "**/id_c.jpg", "**/exp_c.jpg"]

# Store matching files
matching_files = []

# Search for files recursively
for pattern in patterns:
    matching_files.extend(glob.glob(pattern, recursive=True))

# Filter files greater than 300KB
large_files = [file for file in matching_files if os.path.getsize(file) > 300 * 1024]

# Print the count of large files
print(f"Number of images greater than 300KB: {len(large_files)}")


# Compress images larger than 300KB by resizing and optimizing
def compress_image(image_path, max_size=300 * 1024):
    img = Image.open(image_path)
    width, height = img.size

    # Reduce size stepwise and compress
    for scale in [0.9, 0.8, 0.7, 0.6, 0.5]:
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        img_resized.save(image_path, quality=85, optimize=True)

        if os.path.getsize(image_path) <= max_size:
            break


# Compress large images
for file in tqdm(large_files):
    compress_image(file)

# Recount files greater than 300KB after compression
large_files_after = [
    file for file in matching_files if os.path.getsize(file) > 300 * 1024
]
print(
    f"Number of images still greater than 300KB after compression: {len(large_files_after)}"
)
