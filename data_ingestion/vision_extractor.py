import os
import time
import re
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Resolved relative to this file — works regardless of cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_IMAGE_DIR = os.path.join(_PROJECT_ROOT, "data", "extracted_images")


def load_gemini_client() -> genai.Client:
    """
    Loads the API key from the project-root .env and returns an authenticated
    Gemini client. Raises ValueError clearly if the key is missing.
    """
    env_path = os.path.join(_PROJECT_ROOT, ".env")
    load_dotenv(dotenv_path=env_path, override=True)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Add it to your .env file at the project root."
        )

    logger.info("API key loaded (starts with: %s...)", api_key[:5])
    return genai.Client(api_key=api_key)


def describe_images(image_dir: str = DEFAULT_IMAGE_DIR) -> None:
    """
    Sends each extracted chart image through Gemini Vision and saves the
    description as a .txt file alongside the original image.

    Already-processed images are skipped so reruns are safe and cheap.
    Rate-limit (429) and server overload (503) errors are handled with
    automatic retries — other errors abort the current image and move on.
    """
    client = load_gemini_client()
    logger.info("Gemini client ready.")

    valid_extensions = (".png", ".jpg", ".jpeg", ".webp")
    image_files = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)
    )

    if not image_files:
        logger.warning("No images found in %s — nothing to process.", image_dir)
        return

    total = len(image_files)
    logger.info("Found %d chart(s). Starting vision extraction...", total)

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

        if os.path.exists(txt_path):
            logger.info("[%d/%d] Skipping %s (already done).", idx + 1, total, img_name)
            continue

        logger.info("[%d/%d] Analyzing: %s", idx + 1, total, img_name)
        server_errors = 0

        while True:
            try:
                with open(img_path, "rb") as f:
                    image_bytes = f.read()

                response = client.models.generate_content(
model="gemini-3.5-flash-lite",
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        analysis_prompt,
                    ]
                )

                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(response.text)

                logger.info("  Saved → %s", os.path.basename(txt_path))
                time.sleep(6)  # Polite pause to stay within API rate limits
                break

            except Exception as e:
                err = str(e)

                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    match = re.search(r'Please retry in (\d+\.\d+)s', err)
                    wait = float(match.group(1)) + 2.0 if match else 60.0
                    logger.warning("  Rate limited — waiting %.1fs then retrying...", wait)
                    time.sleep(wait)

                elif "503" in err or "UNAVAILABLE" in err:
                    server_errors += 1
                    if server_errors >= 3:
                        logger.error(
                            "  Server rejected %s 3 times — skipping to avoid a hang.", img_name
                        )
                        break
                    logger.warning(
                        "  Server overloaded (attempt %d/3) — waiting 30s...", server_errors
                    )
                    time.sleep(30)

                else:
                    logger.error("  Unexpected error on %s: %s", img_name, err)
                    time.sleep(10)
                    break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    describe_images()
    logger.info("Vision extraction complete.")