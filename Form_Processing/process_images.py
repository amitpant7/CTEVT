import cv2
import os
import sys
import numpy as np
from tqdm import tqdm
import mediapipe as mp


#defining the parametes 
max_size_form = 300 # in kib
max_size_id = 300 #
max_size_exp = 300
rotate = True      #rotate twice or not
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')   #face detection algo


## Naming conventions:
## use id.jpg for citizenship, form.jpg for form and exp.jpg for experience.


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
        aligned_image = cv2.warpAffine(image, rotation_matrix, (w, h),  borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)) 

        return aligned_image
    
    else:
        print("No face detected!")
        return image


def extract_photo(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=100, threshold2=200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    face = False

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        gray_photo = gray[y:y+h, x:x+w]
        faces = face_cascade.detectMultiScale(gray_photo, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
        
        if len(faces)>=1 and (200 < w < 500 and 200 < h < 500):
            passport_photo = image[y:y+h, x:x+w]
            passport_photo = cv2.resize(passport_photo, (192, 192),interpolation=cv2.INTER_LANCZOS4)
            face = True
            break

    if not face:
        passport_photo = np.random.randint(0, 10, (192, 192))

    passport_photo = align_face(passport_photo)
    cv2.imwrite('photo.jpg', passport_photo)
    
    
    

def compress_image(image, output_path, max_size_kb, initial_quality=90, step=5):
    """
    Compress an image to a specified maximum file size in KB.

    """

    quality = initial_quality
    while quality > 0:

        cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])


        current_size_kb = os.path.getsize(output_path) / 1024
        
        if current_size_kb <= max_size_kb:
            return True


        quality -= step

    return False


def process_folder(input_path = '.'):
    
    all_files = []
    for root, dirs, files in os.walk(input_path): 
        for file in files:
            if file.lower().endswith('.jpg'):
                all_files.append(os.path.join(root, file))
            
            
    for p in tqdm(all_files, desc="Processing images"):
            img = cv2.imread(p)
             
            if rotate:
                img = cv2.rotate(img, cv2.ROTATE_180)
            
            if p.endswith('form.jpg'): 
                form_img = img
                ##extract person photo 
                extract_photo(form_img)
                
                ## compress and save the form
                
                compress_path = os.path.join(root, 'form_c.jpg')
                
                sucess = compress_image(form_img, compress_path, max_size_kb=max_size_form)
                
                if not sucess:
                    print('Form compression failed for ', {root}, ' perfrom Manually')
                    
                    
            
            if p.endswith('id.jpg'):
                compress_path = os.path.join(root, 'id_c.jpg')
                sucess = compress_image(form_img, compress_path, max_size_kb=max_size_id)
                
                
                        
            if p.endswith('exp.jpg'):
                compress_path = os.path.join(root, 'exp_c.jpg')
                sucess = compress_image(form_img, compress_path, max_size_kb=max_size_exp)
                
                
                

if __name__ == "__main__":
    args = sys.argv
    
    if len(args)>1:
        print('-----------Starting Image compression and Extraction------------------')
        dir_path = args[1]
        process_folder(args)
        print('----------------Compression and Extraction compelete---------------')
    
    
                
             
                    
                    
            

            
        