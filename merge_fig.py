import os
import glob

try:
    import fitz  # PyMuPDF
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Required libraries missing. Please run: pip install PyMuPDF pillow")
    exit()

# Configuration paths and variables
BASE_DIR = "/home/ubuntu/llm_effective_depth/output"
COMP_OUTPUT_DIR = "comp_output"
MODELS = ["llama_3_1_8b_instruct", "qwen3_8b", "allam_7b_instruct"]
SIZES = ["1", "16", "40"]
EXPERIMENTS = ["layer_effect", "KL"]

# Create the output directory if it doesn't exist
os.makedirs(COMP_OUTPUT_DIR, exist_ok=True)

def pdf_page_to_image(pdf_path):
    """Converts the first page of a PDF to a PIL Image."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    # 200 dpi ensures the stitched image remains high quality and readable
    pix = page.get_pixmap(dpi=200) 
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

def get_font(size):
    """Attempts to load a standard system font, falling back to default if unavailable."""
    try:
        # Standard font path on Ubuntu
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except IOError:
        try:
            # Fallback for Mac/Windows just in case
            return ImageFont.truetype("arial.ttf", size)
        except IOError:
            # Absolute fallback (might look small, but prevents crashing)
            return ImageFont.load_default()

def main():
    print(f"Creating comparison figures in: {os.path.abspath(COMP_OUTPUT_DIR)}\n")
    
    # Define margins for text
    TOP_MARGIN = 150
    BOTTOM_MARGIN = 100
    
    # Load fonts (adjusted sizes for 200 DPI)
    title_font = get_font(60)
    label_font = get_font(45)
    
    for model in MODELS:
        for size in SIZES:
            for exp in EXPERIMENTS:
                # Construct exact paths
                ar_path = os.path.join(BASE_DIR, model, "arabic", f"random{size}PM", exp)
                en_path = os.path.join(BASE_DIR, model, "english", f"random{size}PM", exp)
                
                # Find all PDFs in the Arabic folder
                ar_pdfs = glob.glob(os.path.join(ar_path, "*.pdf"))
                
                if not ar_pdfs:
                    continue
                    
                for ar_pdf_file in ar_pdfs:
                    filename = os.path.basename(ar_pdf_file)
                    en_filename = filename.replace("arabic", "english")
                    en_pdf_file = os.path.join(en_path, en_filename)
                    
                    if not os.path.exists(en_pdf_file):
                        print(f"  [!] Missing matching English PDF for: {filename} in {en_path}")
                        continue
                        
                    print(f"Merging: {model} | MIXED{size} | {exp} -> {filename}")
                    
                    # Convert both PDFs to Images
                    img_ar = pdf_page_to_image(ar_pdf_file)
                    img_en = pdf_page_to_image(en_pdf_file)
                    
                    # Calculate dimensions for the new canvas
                    total_width = img_ar.width + img_en.width
                    max_height = max(img_ar.height, img_en.height)
                    canvas_height = max_height + TOP_MARGIN + BOTTOM_MARGIN
                    
                    # Create a new blank image (white background)
                    combined_img = Image.new('RGB', (total_width, canvas_height), (255, 255, 255))
                    
                    # Paste the images side-by-side (leaving room at top and bottom)
                    combined_img.paste(img_ar, (0, TOP_MARGIN))
                    combined_img.paste(img_en, (img_ar.width, TOP_MARGIN))
                    
                    # Initialize ImageDraw to add text
                    draw = ImageDraw.Draw(combined_img)
                    
                    # --- ADD TOP TITLE (Model Name) ---
                    # Format title text to look clean
                    exp_formatted = exp.replace("_", " ").title()
                    title_text = f"Model: {model}  |  {exp_formatted} (MIXED {size})"
                    
                    # Calculate text size and center it
                    bbox = draw.textbbox((0, 0), title_text, font=title_font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    draw.text(((total_width - text_w) / 2, (TOP_MARGIN - text_h) / 2), 
                              title_text, fill="black", font=title_font)
                    
                    # --- ADD BOTTOM LABELS (Arabic and English) ---
                    # Arabic Label (Centered under the left image)
                    ar_text = "Arabic"
                    bbox = draw.textbbox((0, 0), ar_text, font=label_font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    draw.text(((img_ar.width - text_w) / 2, TOP_MARGIN + max_height + (BOTTOM_MARGIN - text_h) / 2), 
                              ar_text, fill="black", font=label_font)
                    
                    # English Label (Centered under the right image)
                    en_text = "English"
                    bbox = draw.textbbox((0, 0), en_text, font=label_font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    draw.text((img_ar.width + (img_en.width - text_w) / 2, TOP_MARGIN + max_height + (BOTTOM_MARGIN - text_h) / 2), 
                              en_text, fill="black", font=label_font)
                    
                    # Clean up filename for the output and save
                    safe_filename = filename.replace('.pdf', '')
                    out_name = f"comp_{model}_MIXED{size}_{exp}_{safe_filename}.png"
                    out_filepath = os.path.join(COMP_OUTPUT_DIR, out_name)
                    
                    combined_img.save(out_filepath)
                    print(f"  -> Saved: {out_name}\n")

if __name__ == "__main__":
    main()
    print("All comparison figures generated successfully!")