import os
import sys
import re
import time
import mimetypes
from dotenv import load_dotenv
from google import genai
from google.genai import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import IMAGE_DIR, VISION_MODEL, get_api_key

PROMPT = (
    "You are an aerospace data analyst. Describe this chart from a satellite "
    "navigation research paper. Include:\n"
    "1. The chart title (if visible)\n"
    "2. X-axis and Y-axis labels with units\n"
    "3. What the lines, curves or data points show\n"
    "4. The main takeaway of the chart"
)


def describe_images(image_dir=IMAGE_DIR):
    # Writes a .txt description next to every image. Images that already
    # have a .txt are skipped, so the script can be re-run after a failure.
    load_dotenv()
    api_key = get_api_key()
    if not api_key:
        print("No API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file.")
        return
    client = genai.Client(api_key=api_key)

    image_files = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )
    if not image_files:
        print(f"No images found in {image_dir}")
        return

    total = len(image_files)
    print(f"Found {total} image(s)")

    for i, img_name in enumerate(image_files, start=1):
        img_path = os.path.join(image_dir, img_name)
        txt_path = os.path.splitext(img_path)[0] + ".txt"

        if os.path.exists(txt_path):
            print(f"[{i}/{total}] Skipping {img_name} (already done)")
            continue

        print(f"[{i}/{total}] Analyzing {img_name}")
        with open(img_path, "rb") as f:
            image_bytes = f.read()
        mime_type = mimetypes.guess_type(img_path)[0] or "image/jpeg"

        server_errors = 0
        while True:
            try:
                response = client.models.generate_content(
                    model=VISION_MODEL,
                    contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), PROMPT]
                )
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(response.text)
                print(f"  Saved {os.path.basename(txt_path)}")
                time.sleep(6)  # stay under the free tier rate limit
                break

            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    # The error message usually says how long to wait
                    match = re.search(r'Please retry in (\d+\.\d+)s', err)
                    wait = float(match.group(1)) + 2 if match else 60
                    print(f"  Rate limited, retrying in {wait:.0f}s")
                    time.sleep(wait)
                elif "503" in err or "UNAVAILABLE" in err:
                    server_errors += 1
                    if server_errors >= 3:
                        print(f"  Server busy 3 times, skipping {img_name}")
                        break
                    print(f"  Server busy (attempt {server_errors}/3), retrying in 30s")
                    time.sleep(30)
                else:
                    print(f"  Error on {img_name}: {err}")
                    break


if __name__ == "__main__":
    describe_images()
    print("Done")
