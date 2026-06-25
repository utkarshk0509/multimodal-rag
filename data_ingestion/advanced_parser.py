import os
import fitz  # PyMuPDF
import json  # NEW: Required for exporting our text chunks

def parse_satellite_pdf(file_path):
    print(f"🚀 Analyzing document structure natively: {file_path}")
    
    # Create a folder to save our extracted diagrams!
    image_out_dir = "data/extracted_images"
    if not os.path.exists(image_out_dir):
        os.makedirs(image_out_dir)

    try:
        doc = fitz.open(file_path)
        print(f"📄 Successfully loaded {len(doc)} pages.\n")
        
        total_text_blocks = 0
        total_tables = 0
        total_images = 0
        
        # NEW: We need an empty list to actually hold our text strings!
        extracted_chunks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            print(f"--- Processing Page {page_num + 1} ---")
            
            # 1. Detect and Extract Tables (BUG FIXED)
            table_finder = page.find_tables()
            page_tables = table_finder.tables  # We add .tables to get the actual list!
            page_tables_count = len(page_tables)
            total_tables += page_tables_count
            
            if page_tables_count > 0:
                print(f"  📊 Found {page_tables_count} table(s):")
                for idx, table in enumerate(page_tables):
                    data = table.extract()
                    print(f"    - Table [{idx}]: Header -> {data[0] if data else 'Empty'}")
            
            # 2. Extract Text Blocks
            blocks = page.get_text("blocks")
            page_text_blocks = 0
            for block in blocks:
                text_content = block[4].strip()
                if len(text_content) > 10:
                    page_text_blocks += 1
                    total_text_blocks += 1
                    
                    # NEW: Append the actual text content to our list
                    extracted_chunks.append(text_content)
                    
            print(f"  📝 Extracted {page_text_blocks} narrative text blocks.")

            # 3. Detect and Extract Images (NEW LOGIC)
            image_list = page.get_images(full=True)
            page_image_count = len(image_list)
            total_images += page_image_count

            if page_image_count > 0:
                print(f"  🖼️ Found {page_image_count} image(s)/diagram(s):")
                for img_index, img in enumerate(image_list):
                    xref = img[0] # The internal PDF reference ID for the image
                    
                    # Extract the raw image bytes
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Create a clean filename and save it
                    image_filename = f"page{page_num+1}_img{img_index+1}.{image_ext}"
                    image_filepath = os.path.join(image_out_dir, image_filename)
                    
                    with open(image_filepath, "wb") as image_file:
                        image_file.write(image_bytes)
                    print(f"    - Saved diagram as: {image_filename}")
            
            print("") # Just printing an empty line to keep the terminal readable
            
        print("=========================================")
        print("🎉 PARSING COMPLETE SUMMARY:")
        print(f"   - Total Structured Text Chunks: {total_text_blocks}")
        print(f"   - Total Tables Extracted Natively: {total_tables}")
        print(f"   - Total Images/Diagrams Saved: {total_images}")
        print("=========================================")
        
        # NEW: Save the extracted chunks to a JSON file
        output_path = os.path.join("data", "narrative_chunks.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extracted_chunks, f, ensure_ascii=False, indent=4)
            
        print(f"✅ Success! Saved {len(extracted_chunks)} chunks to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error parsing the document: {e}")

if __name__ == "__main__":
    # I noticed your file in the screenshot was capitalized as SAMPLE.pdf
    pdf_location = "data/SAMPLE.pdf" 
    
    if not os.path.exists(pdf_location):
        print(f"❌ Could not find {pdf_location}.")
    else:
        parse_satellite_pdf(pdf_location)