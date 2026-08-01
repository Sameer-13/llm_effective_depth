import os
import glob
from PIL import Image, ImageDraw, ImageFont

# Configuration paths and variables
COMP_OUTPUT_DIR = "comp_output"
PDF_OUTPUT_DIR = "pdf_reports"
MODELS = ["llama_3_1_8b_instruct", "qwen3_8b", "allam_7b_instruct"]

# Excluded "1" as requested
SIZES = ["16", "40"]
EXPERIMENTS = ["layer_effect", "KL"]

# Create the output directory for PDFs
os.makedirs(PDF_OUTPUT_DIR, exist_ok=True)

def get_font(size, bold=False):
    """Attempts to load a standard system font, falling back to default if unavailable."""
    font_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{font_name}", size)
    except IOError:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except IOError:
            return ImageFont.load_default()

def create_cover_page(size, exp, width=2000, height=2500):
    """Creates a simple text-based image to act as a PDF section cover."""
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    title_font = get_font(100, bold=True)
    sub_font = get_font(70)
    
    title_text = f"Dataset Size: MIXED{size}"
    exp_formatted = exp.replace("_", " ").title()
    sub_text = f"Experiment: {exp_formatted}"
    desc_text = "The following pages contain the stacked comparison charts\nfor LLaMA, Qwen, and ALLaM."
    
    # Title
    bbox = draw.textbbox((0, 0), title_text, font=title_font)
    w = bbox[2] - bbox[0]
    draw.text(((width - w) / 2, height / 2 - 200), title_text, fill="black", font=title_font)
    
    # Subtitle
    bbox = draw.textbbox((0, 0), sub_text, font=sub_font)
    w = bbox[2] - bbox[0]
    draw.text(((width - w) / 2, height / 2), sub_text, fill=(80, 80, 80), font=sub_font)

    # Description
    bbox = draw.textbbox((0, 0), desc_text, font=sub_font)
    w = bbox[2] - bbox[0]
    draw.multiline_text(((width - w) / 2, height / 2 + 150), desc_text, fill=(120, 120, 120), font=sub_font, align="center")
    
    return img

def main():
    print(f"Creating master PDF report in: {os.path.abspath(PDF_OUTPUT_DIR)}\n")
    
    master_pdf_pages = []
    
    for size in SIZES:
        for exp in EXPERIMENTS:
            print(f"Processing: MIXED{size} | {exp}")
            
            # 1. Find all base files using the first model as a reference
            ref_model = MODELS[0]
            search_pattern = os.path.join(COMP_OUTPUT_DIR, f"comp_{ref_model}_MIXED{size}_{exp}_*.png")
            ref_files = glob.glob(search_pattern)
            
            if not ref_files:
                print(f"  [!] No images found for MIXED{size} {exp}. Skipping...\n")
                continue
            
            # Insert the cover page for this section before processing the charts
            cover_page = create_cover_page(size, exp)
            master_pdf_pages.append(cover_page)
            
            # 2. For each unique chart type, stack the 3 models
            for ref_file in ref_files:
                base_name = os.path.basename(ref_file)
                
                # Extract the universal suffix
                prefix_with_model = f"comp_{ref_model}_MIXED{size}_{exp}_{ref_model}"
                if not base_name.startswith(prefix_with_model):
                    continue
                universal_suffix = base_name[len(prefix_with_model):] 
                
                images_to_stack = []
                
                # Gather the comparison image for each model
                for model in MODELS:
                    img_name = f"comp_{model}_MIXED{size}_{exp}_{model}{universal_suffix}"
                    img_path = os.path.join(COMP_OUTPUT_DIR, img_name)
                    
                    if os.path.exists(img_path):
                        images_to_stack.append(Image.open(img_path))
                    else:
                        print(f"  [!] Warning: Missing image {img_name}")
                
                if not images_to_stack:
                    continue
                
                # 3. Calculate canvas dimensions for the vertical stack
                max_width = max(img.width for img in images_to_stack)
                total_height = sum(img.height for img in images_to_stack)
                
                # Create a blank white canvas
                stacked_page = Image.new('RGB', (max_width, total_height), (255, 255, 255))
                
                # 4. Paste images sequentially from top to bottom
                current_y = 0
                for img in images_to_stack:
                    x_offset = (max_width - img.width) // 2
                    stacked_page.paste(img, (x_offset, current_y))
                    current_y += img.height
                
                master_pdf_pages.append(stacked_page)
                
    # 5. Save everything into ONE master PDF
    if master_pdf_pages:
        pdf_filename = "Master_Comparison_Report.pdf"
        pdf_path = os.path.join(PDF_OUTPUT_DIR, pdf_filename)
        
        master_pdf_pages[0].save(
            pdf_path, 
            "PDF", 
            resolution=100.0, 
            save_all=True, 
            append_images=master_pdf_pages[1:]
        )
        print(f"============================================================")
        print(f"SUCCESS: Saved {pdf_filename} ({len(master_pdf_pages)} total pages)")
        print(f"============================================================")

if __name__ == "__main__":
    main()