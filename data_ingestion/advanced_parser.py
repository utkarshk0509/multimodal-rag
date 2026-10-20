import os
import fitz   # PyMuPDF
import json


def parse_satellite_pdf(file_path):
    """
    Extracts text blocks, tables, and images from a satellite research PDF.
    Images are saved to disk; text is collected and dumped to a JSON file
    for the next stage of the pipeline.
    """
    print(f"Opening document: {file_path}")

    image_out_dir = "data/extracted_images"
    os.makedirs(image_out_dir, exist_ok=True)

    try:
        doc = fitz.open(file_path)
        print(f"Loaded {len(doc)} pages.\n")

        text_block_count = 0
        table_count = 0
        image_count = 0
        extracted_chunks = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            print(f"--- Page {page_num + 1} ---")

            # --- Tables ---
            found_tables = page.find_tables().tables
            table_count += len(found_tables)

            if found_tables:
                print(f"  Found {len(found_tables)} table(s):")
                for idx, table in enumerate(found_tables):
                    data = table.extract()
                    header = data[0] if data else "Empty"
                    print(f"    Table [{idx}]: {header}")

            # --- Text blocks ---
            # Only keep blocks with meaningful content (skip labels, headers < 10 chars)
            for block in page.get_text("blocks"):
                text = block[4].strip()
                if len(text) > 10:
                    text_block_count += 1
                    extracted_chunks.append(text)

            print(f"  Extracted {text_block_count} text blocks so far.")

            # --- Images / Diagrams ---
            images = page.get_images(full=True)
            image_count += len(images)

            if images:
                print(f"  Found {len(images)} image(s):")
                for img_idx, img in enumerate(images):
                    xref = img[0]
                    raw = doc.extract_image(xref)
                    img_filename = f"page{page_num+1}_img{img_idx+1}.{raw['ext']}"
                    img_path = os.path.join(image_out_dir, img_filename)

                    with open(img_path, "wb") as f:
                        f.write(raw["image"])
                    print(f"    Saved: {img_filename}")

            print("")

        # Summary
        print("=========================================")
        print("Parse complete:")
        print(f"  Text chunks : {text_block_count}")
        print(f"  Tables      : {table_count}")
        print(f"  Images      : {image_count}")
        print("=========================================")

        # Dump text chunks to JSON for the vector store ingestion step
        output_path = os.path.join("data", "narrative_chunks.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extracted_chunks, f, ensure_ascii=False, indent=4)

        print(f"Saved {len(extracted_chunks)} chunks → {output_path}")

    except Exception as e:
        print(f"Failed to parse document: {e}")


if __name__ == "__main__":
    pdf_location = "data/SAMPLE.pdf"

    if not os.path.exists(pdf_location):
        print(f"File not found: {pdf_location}")
    else:
        parse_satellite_pdf(pdf_location)