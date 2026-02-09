import os
from pdf2image import convert_from_path

def process_pdfs(directory):
    # Recursively find all PDF files
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))

    if not pdf_files:
        print("No PDF files found.")
        return

    image_count = 1  # Counter for multi-page images

    for pdf_path in pdf_files:
        images = convert_from_path(pdf_path)
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]  # Get filename without extension
        folder = os.path.dirname(pdf_path)  # Get folder of the PDF

        if len(images) == 1:
            # Single-page PDF: Save as <PDFNAME>.jpg
            jpg_path = os.path.join(folder, f"{base_name}.jpg")
            images[0].save(jpg_path, "JPEG")
            print(f"Converted {pdf_path} to {jpg_path}")
        else:
            # Multi-page PDF: Save pages as Image_1.jpg, Image_2.jpg, etc.
            for img in images:
                img_path = os.path.join(folder, f"Image_{image_count}.jpg")
                img.save(img_path, "JPEG")
                image_count += 1
            print(f"Extracted {len(images)} pages from {pdf_path}")

        os.remove(pdf_path)  # Delete the original PDF
        print(f"Removed {pdf_path}")

directory = "/path/to/your/pdf/folder"  # Change this to your actual folder path
process_pdfs(directory)
