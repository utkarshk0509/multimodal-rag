import fitz 

def test_read_pdf(file_path):
    print(f"Attempting to open: {file_path}...")
    
    try:
        doc = fitz.open(file_path)
        print(f"The document has {len(doc)} pages.")
        
        first_page_text = doc[0].get_text()
        
        print("\n--- PREVIEW OF PAGE 1 ---")
        print(first_page_text[:500])
        print("-------------------------\n")
        
    except Exception as e:
        print(f"Error reading the PDF: {e}")

if __name__ == "__main__":
    pdf_location = "data/SAMPLE.pdf"
    test_read_pdf(pdf_location)