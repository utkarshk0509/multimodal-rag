import os
import time
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types


def load_gemini_client():
    """
    Loads the API key from the project root .env and returns an authenticated
    Gemini client. Raises clearly if the key is missing so you don't get a
    cryptic error later.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    env_path = os.path.join(project_root, ".env")

    load_dotenv(dotenv_path=env_path, override=True)
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing — check your .env file.")

    print(f"API key loaded (starts with: {api_key[:5]}...)")
    return genai.Client(api_key=api_key)


def describe_images(image_dir):
    """
    Sends each extracted chart image through Gemini Vision and saves the
    description as a .txt file alongside the image.

    Already-processed images are skipped so reruns are safe and cheap.
    Rate limit and server errors are handled with automatic retries.
    """
    client = load_gemini_client()
    print("Gemini client ready.\n")

    valid_extensions = (".png", ".jpg", ".jpeg", ".webp")
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print("No images found in the directory — nothing to process.")
        return

    total = len(image_files)
    print(f"Found {total} chart(s). Starting vision extraction...\n")

    # Tell the model exactly what we need: title, axes, trends, and the key takeaway
    analysis_prompt = (
        "You are an expert aerospace data analyst. Analyze this chart from a satellite "
        "navigation research paper and give a structured, detailed description. Include:\n"
        "1. The chart title or header (if visible)\n"
        "2. X-axis and Y-axis labels with units\n"
        "3. What the trend lines, curves, or data clusters show\n"
        "4. The core engineering insight the chart communicates"
    )

    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(image_dir, img_name)
        txt_path = os.path.splitext(img_path)[0] + ".txt"

        # Skip if we already have a description for this image
        if os.path.exists(txt_path):
            print(f"[{idx+1}/{total}] Skipping {img_name} (already done).")
            continue

        print(f"[{idx+1}/{total}] Analyzing: {img_name}...")
        server_errors = 0

        while True:
            try:
                with open(img_path, "rb") as f:
                    image_bytes = f.read()

                response = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        analysis_prompt
                    ]
                )

                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(response.text)

                print(f"  Saved → {os.path.basename(txt_path)}")
                time.sleep(6)  # Stay polite to the API rate limits
                break

            except Exception as e:
                err = str(e)

                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    # Try to extract the exact retry delay from the error message
                    match = re.search(r'Please retry in (\d+\.\d+)s', err)
                    if match:
                        wait = float(match.group(1)) + 2.0
                        print(f"  Rate limited — waiting {wait:.1f}s then retrying...")
                    else:
                        wait = 60
                        print(f"  Rate limited — waiting {wait}s then retrying...")
                    time.sleep(wait)

                elif "503" in err or "UNAVAILABLE" in err:
                    server_errors += 1
                    if server_errors >= 3:
                        print(f"  Server rejected {img_name} 3 times — skipping to avoid getting stuck.")
                        break
                    print(f"  Server overloaded (attempt {server_errors}/3) — waiting 30s...")
                    time.sleep(30)

                else:
                    print(f"  Unexpected error on {img_name}: {err}")
                    time.sleep(10)
                    break


if __name__ == "__main__":
    target_folder = "data/extracted_images"
    describe_images(target_folder)
    print("\nVision extraction complete.")