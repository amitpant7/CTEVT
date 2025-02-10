import cv2
import glob

size = (192, 192)
input_dir = 'RPL_processed1'
all_person_photos = glob.glob(f'{input_dir}/**/zzz_backup.jpg', recursive=True)

for path in all_person_photos:
    image = cv2.imread(path)

    if image is None:
        print(f"Warning: Could not read image {path}")
        continue  # Skip if image couldn't be read
    
    image = cv2.resize(image, size)

    cv2.imwrite(path, image)  # Corrected order
