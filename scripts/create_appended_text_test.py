from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
TEST_DIR = BASE_DIR / "tests" / "test_images"
SOURCE_IMAGE = TEST_DIR / "clean.png"
OUTPUT_IMAGE = TEST_DIR / "appended_text.png"

PAYLOAD = b"""
=== HIDDEN FORENSIC TEST PAYLOAD ===
This is a controlled payload for testing the AI Multimedia Steganography Detection system.
Technique: Appended Data
Payload Type: Plain Text
Purpose: Verify detection and extraction.
=== END PAYLOAD ===
"""


def main():
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    if not SOURCE_IMAGE.exists():
        raise FileNotFoundError(
            f"Clean test image not found: {SOURCE_IMAGE}\n"
            "Create clean.png first."
        )

    image_data = SOURCE_IMAGE.read_bytes()
    OUTPUT_IMAGE.write_bytes(image_data + PAYLOAD)

    print(f"Created: {OUTPUT_IMAGE}")
    print(f"Original size : {len(image_data)} bytes")
    print(f"Payload size  : {len(PAYLOAD)} bytes")
    print(f"Final size    : {OUTPUT_IMAGE.stat().st_size} bytes")


if __name__ == "__main__":
    main()