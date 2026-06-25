import os
import time
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types

def initialize_gemini_client():
    """Initializes the Gemini developer client using the absolute environment key path."""
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    root_project_dir = os.path.dirname(current_script_dir)
    exact_env_path = os.path.join(root_project_dir, ".env")
    
    load_dotenv(dotenv_path=exact_env_path, override=True)
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("[CRITICAL] API Key is completely empty. Check your .env file.")
        
    print(f"[INFO] Key successfully loaded. (Starts with: {api_key[:5]}...)")
    return genai.Client(api_key=api_key)

def analyze_all_extracted_images(image_dir):
    """Loops through all images, generates an AI description, and handles dynamic rate limits."""
    client = initialize_gemini_client()
    print("[INFO] Gemini Client successfully authenticated.")
    
    valid_extensions = (".png", ".jpg", ".jpeg", ".webp")
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print("[WARNING] No images found in the directory.")
        return

    total_images = len(image_files)
    print(f"[INFO] Found {total_images} charts. Resuming extraction pipeline...\n")
    
    system_instruction = (
        "You are an expert aerospace data analyst. Analyze this chart extracted from a satellite "
        "navigation research paper. Provide a structured, highly granular text description. "
        "Include: 1. The explicit Title or Header if visible. 2. The X-axis and Y-axis labels and units. "
        "3. A breakdown of the trend lines, curves, or data clusters shown. "
        "4. The core engineering conclusion represented by the visualization."
    )

    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(image_dir, img_name)
        txt_output_path = os.path.splitext(img_path)[0] + ".txt"
        
        if os.path.exists(txt_output_path):
            print(f"[SKIP] [{idx+1}/{total_images}]: {img_name} (Already processed).")
            continue
            
        print(f"[PROCESS] Analyzing [{idx+1}/{total_images}]: {img_name}...")
        
        # Track consecutive 503 server errors for this specific image
        server_error_count = 0
        
        while True:
            try:
                with open(img_path, "rb") as img_file:
                    image_bytes = img_file.read()
                    
                response = client.models.generate_content(
                    model="gemini-2.5-flash-lite", 
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        system_instruction
                    ]
                )
                
                with open(txt_output_path, "w", encoding="utf-8") as txt_file:
                    txt_file.write(response.text)
                    
                print(f"  [SUCCESS] Saved description to: {os.path.basename(txt_output_path)}")
                time.sleep(6) 
                break 
                
            except Exception as e:
                error_msg = str(e)
                
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    match = re.search(r'Please retry in (\d+\.\d+)s', error_msg)
                    if match:
                        exact_cooldown = float(match.group(1)) + 2.0 
                        print(f"  [RATE LIMIT] 429 Error. Pausing for {exact_cooldown:.1f} seconds...")
                        time.sleep(exact_cooldown)
                    else:
                        print("  [RATE LIMIT] 429 Error. Pausing for 60 seconds...")
                        time.sleep(60)
                    print("  [RETRY] Retrying the current image...")
                    
                elif "503" in error_msg or "UNAVAILABLE" in error_msg:
                    server_error_count += 1
                    if server_error_count >= 3:
                        print(f"  [BYPASS] Server rejected {img_name} 3 times. Moving to next image to prevent freezing.")
                        break # Exits the while loop, allowing the for loop to grab the next image
                    else:
                        print(f"  [SERVER ERROR] 503 Overloaded (Attempt {server_error_count}/3). Pausing 30s...")
                        time.sleep(30)
                    
                else:
                    print(f"  [ERROR] Unrecoverable error on {img_name}: {error_msg}")
                    time.sleep(10)
                    break 

if __name__ == "__main__":
    target_folder = "data/extracted_images"
    analyze_all_extracted_images(target_folder)
    print("\n[PHASE 2 COMPLETE] Vision pipeline execution finished.")