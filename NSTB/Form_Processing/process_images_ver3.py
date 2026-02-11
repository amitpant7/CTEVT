import warnings
warnings.filterwarnings("ignore")

import cv2
import os
import sys
import numpy as np
from tqdm import tqdm
import mediapipe as mp
import warnings
from PIL import Image
from ultralytics import YOLO


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress TensorFlow logs

# For MediaPipe to use CPU only
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
cv2.setLogLevel(0)  # Suppress OpenCV warnings

# Previous parameters remain the same...
max_size_form = 300 
max_size_id = 300
max_size_exp = 300
rotate = False
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

warnings.filterwarnings("ignore", category=UserWarning)

def align_face(image):
    """
    Enhanced face alignment function that ensures vertical alignment and proper cropping
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    # Convert to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return image

    landmarks = results.multi_face_landmarks[0]
    h, w = image.shape[:2]

    # Get key facial landmarks
    # Use nose bridge top and bottom for vertical alignment
    nose_top = landmarks.landmark[168]  # Top of nose bridge
    nose_bottom = landmarks.landmark[6]  # Bottom of nose
    left_eye = landmarks.landmark[33]    # Left eye inner corner
    right_eye = landmarks.landmark[263]  # Right eye inner corner

    # Convert landmarks to pixel coordinates
    nose_top = np.array([nose_top.x * w, nose_top.y * h])
    nose_bottom = np.array([nose_bottom.x * w, nose_bottom.y * h])
    left_eye = np.array([left_eye.x * w, left_eye.y * h])
    right_eye = np.array([right_eye.x * w, right_eye.y * h])

    # Calculate angles for both horizontal and vertical alignment
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle_horizontal = np.degrees(np.arctan2(dy, dx))

    dx_vertical = nose_bottom[0] - nose_top[0]
    dy_vertical = nose_bottom[1] - nose_top[1]
    angle_vertical = np.degrees(np.arctan2(dx_vertical, dy_vertical))

    # Combine the angles
    angle = angle_horizontal

    # If face is significantly tilted vertically (more than 45 degrees)
    if abs(angle_vertical) > 45:
        angle += 90 * np.sign(angle_vertical)

    # Create a rotation matrix
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new image dimensions to prevent cropping
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust translation part of the matrix
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # Perform the rotation with white background
    aligned_image = cv2.warpAffine(
        image, 
        rotation_matrix, 
        (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    # Find the face in the aligned image
    gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Get the largest face
        x, y, w, h = max(faces, key=lambda x: x[2] * x[3])
        
        # Add padding
        padding_x = int(w * 0.3)
        padding_y = int(h * 0.4)  # More vertical padding for passport-style photo
        
        # Calculate new coordinates with padding
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(aligned_image.shape[1], x + w + padding_x)
        y2 = min(aligned_image.shape[0], y + h + padding_y)
        
        # Crop the face region with padding
        face_crop = aligned_image[y1:y2, x1:x2]
        
        # Resize to maintain aspect ratio
        target_height = 192
        aspect_ratio = face_crop.shape[1] / face_crop.shape[0]
        target_width = int(target_height * aspect_ratio)
        
        # Resize the image
        resized = cv2.resize(face_crop, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Create a white canvas of target size
        final_image = np.ones((192, 192, 3), dtype=np.uint8) * 255
        
        # Calculate positioning to center the face
        x_offset = (192 - target_width) // 2
        
        try:
            final_image[:, x_offset:x_offset+target_width] = resized
            
        except:
            print('Align..., Failed for one Image.')
        
        return final_image

    return cv2.resize(aligned_image, (192, 192), interpolation=cv2.INTER_LANCZOS4)

# The rest of the code remains the same...
class PhotoExtractor:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        
    def extract_photos(self, image):
        # Previous implementation remains the same...
        results = self.model(image, classes=[0], conf=0.2)
        
        if len(results) == 0 or len(results[0].boxes.xyxy) == 0:
            return False, np.random.randint(0, 10, (192, 192, 3)), np.random.randint(0, 10, (192, 192, 3))
            
        box = results[0].boxes.xyxy[0]
        x1, y1, x2, y2 = map(int, box.tolist())
        
        width = x2 - x1
        height = y2 - y1
        
        img_h, img_w = image.shape[:2]
        padded_x1 = max(0, x1 - int(width * 0.15))
        padded_y1 = max(0, y1 - int(height * 0.15))
        padded_x2 = min(img_w, x2 + int(width * 0.15))
        padded_y2 = min(img_h, y2 + int(height * 0.15))
        
        main_x1 = max(0, x1 - int(width * 0.03))
        main_y1 = max(0, y1 - int(height * 0.03))
        main_x2 = min(img_w, x2 + int(width * 0.03))
        main_y2 = min(img_h, y2 + int(height * 0.03))
        
        backup_photo = image[padded_y1:padded_y2, padded_x1:padded_x2]
        main_photo = image[main_y1:main_y2, main_x1:main_x2]
        
        gray_photo = cv2.cvtColor(main_photo, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_photo, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
        
        if len(faces) >= 1:
            main_photo = align_face(main_photo)
            backup_photo = cv2.resize(backup_photo, (192, 192), interpolation=cv2.INTER_LANCZOS4)
            return True, main_photo, backup_photo
        
        return False, np.random.randint(0, 10, (192, 192, 3)), np.random.randint(0, 10, (192, 192, 3))


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
        
        elif p.endswith('id.jpg'):
            compress_path = os.path.join(parent, 'id_c.jpg')
            compress_image(img, compress_path, max_size_kb=max_size_id)
        
        elif p.endswith('exp.jpg'):
            compress_path = os.path.join(parent, 'exp_c.jpg')
            compress_image(img, compress_path, max_size_kb=max_size_exp)
            
        elif not p.endswith('_c.jpg'):
            compress_path = os.path.join(parent, os.path.basename(p).split('.')[0] + '_c.jpg')
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
        print("Usage: python script.py <input_directory>")