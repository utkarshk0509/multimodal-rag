import os
import sys
import json
import pymupdf

# Allow imports from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DEFAULT_PDF_PATH, IMAGE_DIR, CHUNKS_JSON_PATH


def parse_pdf(file_path, image_dir=IMAGE_DIR, chunks_path=CHUNKS_JSON_PATH):
    # Saves text blocks to a JSON file and every embedded image to image_dir.
    # Tables are only counted for now, their text still comes through the text blocks.
    print(f"Opening {file_path}")
    os.makedirs(image_dir, exist_ok=True)

    doc = pymupdf.open(file_path)
    print(f"Loaded {len(doc)} page(s)")

    chunks = []
    table_count = 0
    image_count = 0

    for page_num in range(len(doc)):
        page = doc[page_num]

        table_count += len(page.find_tables().tables)

        # Skip very short blocks like axis labels and page numbers
        for block in page.get_text("blocks"):
            text = block[4].strip()
            if len(text) > 10:
                chunks.append(text)

        for img_num, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            image = doc.extract_image(xref)
            img_name = f"page{page_num + 1}_img{img_num + 1}.{image['ext']}"
            with open(os.path.join(image_dir, img_name), "wb") as f:
                f.write(image["image"])
            image_count += 1

    doc.close()

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)

    print(f"Found {len(chunks)} text chunks, {table_count} tables, {image_count} images")
    print(f"Text chunks saved to {chunks_path}")
    return chunks


if __name__ == "__main__":
    if not os.path.exists(DEFAULT_PDF_PATH):
        print(f"File not found: {DEFAULT_PDF_PATH}")
    else:
        parse_pdf(DEFAULT_PDF_PATH)
