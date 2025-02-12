import cv2
import os
import sys
import numpy as np
from tqdm import tqdm
import mediapipe as mp
import warnings
from PIL import Image
from ultralytics import YOLO

# Defining the parameters 
max_size_form = 300 # in kib
max_size_id = 300
max_size_exp = 300
rotate = False  # rotate twice or not
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Suppress all warnings of a specific type
warnings.filterwarnings("ignore", category=UserWarning)

class PhotoExtractor:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        
    def extract_photos(self, image):
        # Run detection
        results = self.model(image, classes=[0], conf=0.2)  # Assuming class 0 is person
        
        if len(results) == 0 or len(results[0].boxes.xyxy) == 0:
            return False, np.random.randint(0, 10, (192, 192, 3)), np.random.randint(0, 10, (192, 192, 3))
            
        # Get the first detected box
        box = results[0].boxes.xyxy[0]
        x1, y1, x2, y2 = map(int, box.tolist())
        
        # Calculate dimensions for padding
        width = x2 - x1
        height = y2 - y1
        
        # Calculate padded coordinates for backup photo
        img_h, img_w = image.shape[:2]
        padded_x1 = max(0, x1 - int(width * 0.15))
        padded_y1 = max(0, y1 - int(height * 0.15))
        padded_x2 = min(img_w, x2 + int(width * 0.15))
        padded_y2 = min(img_h, y2 + int(height * 0.15))
        
        # Calculate coordinates for main photo (less padding)
        main_x1 = max(0, x1 - int(width * 0.03))
        main_y1 = max(0, y1 - int(height * 0.03))
        main_x2 = min(img_w, x2 + int(width * 0.03))
        main_y2 = min(img_h, y2 + int(height * 0.03))
        
        # Extract both versions of the photo
        backup_photo = image[padded_y1:padded_y2, padded_x1:padded_x2]
        main_photo = image[main_y1:main_y2, main_x1:main_x2]
        
        # Verify face in the main photo
        gray_photo = cv2.cvtColor(main_photo, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_photo, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
        
        if len(faces) >= 1:
            # Resize main photo to standard size
            main_photo = cv2.resize(main_photo, (192, 192), interpolation=cv2.INTER_LANCZOS4)
            backup_photo = cv2.resize(backup_photo, (192, 192), interpolation=cv2.INTER_LANCZOS4)
            
            # Align face in main photo
            main_photo = align_face(main_photo)
            return True, main_photo, backup_photo
        
        return False, np.random.randint(0, 10, (192, 192, 3)), np.random.randint(0, 10, (192, 192, 3))

def align_face(image):
    # Initialize Mediapipe Face Mesh for facial landmark detection
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        
        # Extract coordinates for left and right eyes
        h, w, _ = image.shape
        left_eye = landmarks.landmark[33]  # Left eye inner corner
        right_eye = landmarks.landmark[263]  # Right eye inner corner
        
        left_eye_coords = (int(left_eye.x * w), int(left_eye.y * h))
        right_eye_coords = (int(right_eye.x * w), int(right_eye.y * h))
        
        # Compute the angle of rotation
        delta_y = right_eye_coords[1] - left_eye_coords[1]
        delta_x = right_eye_coords[0] - left_eye_coords[0]
        angle = np.degrees(np.arctan2(delta_y, delta_x))
        
        # Create a rotation matrix
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate the image
        aligned_image = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))
        return aligned_image
    
    return image

def compress_image(image, output_path, max_size_kb, initial_quality=90, step=5):
    quality = initial_quality
    while quality > 0:
        cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        current_size_kb = os.path.getsize(output_path) / 1024
        
        if current_size_kb <= max_size_kb-10:
            return True
        quality -= step
    return False

def process_folder(input_path, model_path):
    photo_extractor = PhotoExtractor(model_path)
    
    all_files = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                all_files.append(os.path.join(root, file))
    
    for p in tqdm(all_files, desc="Processing images"):
        parent = os.path.dirname(p)
        img = cv2.imread(p)
        
        if rotate:
            img = cv2.rotate(img, cv2.ROTATE_180)
        
        if p.endswith('form.jpg'):
            form_img = img
            # Extract person photo using YOLO
            success, photo, photo2 = photo_extractor.extract_photos(form_img)
            
            if not success:
                print('Could not detect face for:', parent)
            else:
                try:
                    cv2.imwrite(os.path.join(parent, 'photo.jpg'), photo)
                    cv2.imwrite(os.path.join(parent, 'zzz_backup.jpg'), photo2)
                except:
                    pass
            
            # Compress and save the form
            compress_path = os.path.join(parent, 'form_c.jpg')
            success = compress_image(form_img, compress_path, max_size_kb=max_size_form)
            
            if not success:
                print('Form compression failed for', parent, 'perform manually')
        
        if p.endswith('id.jpg'):
            compress_path = os.path.join(parent, 'id_c.jpg')
            compress_image(img, compress_path, max_size_kb=max_size_id)
        
        if p.endswith('exp.jpg'):
            compress_path = os.path.join(parent, 'exp_c.jpg')
            compress_image(img, compress_path, max_size_kb=max_size_exp)

if __name__ == "__main__":
    args = sys.argv
    
    if len(args) > 1:
        print('-----------Starting Image compression and Extraction------------------')
        dir_path = args[1]
        model_path = "yolo11x.pt"
        process_folder(input_path=dir_path, model_path=model_path)
        print('----------------Compression and Extraction complete---------------')
    else:
        print("Usage: python script.py <input_directory> <model_path>")