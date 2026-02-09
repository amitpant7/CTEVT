### README: How to Use `process_images.py`

`process_images.py` is a Python script designed to process images, align facial photos, extract passport-sized photos from forms, and compress images to meet specific file size requirements.

#### Prerequisites

1. **Python Environment**:
   Ensure you have Python 3.6+ installed on your system.
   
2. **Required Libraries**:s
   Install the necessary Python libraries using the following command:
   ```bash
   pip install numpy opencv-python tqdm mediapipe
   ```

3. **Input Images**:
   The script processes images located in a specified directory. The images should follow these naming conventions:
   - **ID Image**: `id.jpg`
   - **Form Image**: `form.jpg`
   - **Experience Image**: `exp.jpg`

   If images don't follow this convention, and follow the ordering during scanning process run `rename.py` to fix the file naming.
   
   ```bash 
   python rename.py individual_folder
   ```

#### Features
- **Image Compression**:
  Compresses images (`id.jpg`, `form.jpg`, `exp.jpg`) to specified file size limits.
- **Face Alignment**:
  Aligns the face in the extracted passport-sized photo using Mediapipe's Face Mesh.
- **Passport Photo Extraction**:
  Detects and extracts the face photo from `form.jpg`, resizes it to 192x192 pixels, and saves it as `photo.jpg`.
- **Image Rotation**:
  Optionally rotates the images during processing.

#### Script Parameters
- `max_size_form`, `max_size_id`, `max_size_exp`: Maximum file sizes (in KB) for the respective images.
- `rotate`: Whether to rotate the images during processing (`True` or `False`).

#### Usage

1. **Script Location**:
   Place the script (`process_images.py`) in your desired directory.

2. **Run the Script**:
   Execute the script by specifying the directory containing the images:
   ```bash
   python process_images.py /path/to/image/directory
   ```

   - Replace `/path/to/image/directory` with the absolute or relative path to the folder containing the images.

3. **Output**:
   - Compressed images will be saved with the `_c` suffix in the same directory.
     - Compressed ID: `id_c.jpg`
     - Compressed Form: `form_c.jpg`
     - Compressed Experience: `exp_c.jpg`
   - Extracted passport-sized photo from `form.jpg` will be saved as `photo.jpg`.

#### Example Directory Structure
```plaintext
input_directory/
    id.jpg
    form.jpg
    exp.jpg
```

#### Notes
- If the script cannot compress an image below the specified size, it will notify you to compress it manually.
- If no face is detected for the passport photo, a random placeholder image is generated.