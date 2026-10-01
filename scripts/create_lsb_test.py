from pathlib import Path

from PIL import Image


BASE_DIR = Path(__file__).resolve().parents[1]
TEST_DIR = BASE_DIR / "tests" / "test_images"

SOURCE_IMAGE = TEST_DIR / "clean.png"
OUTPUT_IMAGE = TEST_DIR / "lsb_text.png"

SECRET_MESSAGE = (
    "STEGO FORENSIC TEST: This message is hidden inside the image "
    "using least-significant-bit substitution."
)


def bytes_to_bits(data: bytes) -> list[int]:
    bits = []

    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)

    return bits


def main():
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    if not SOURCE_IMAGE.exists():
        raise FileNotFoundError(
            f"Clean image not found:\n{SOURCE_IMAGE}"
        )

    image = Image.open(SOURCE_IMAGE).convert("RGB")
    pixels = list(image.getdata())

    message_bytes = SECRET_MESSAGE.encode("utf-8")

    # Store the message length first.
    payload = len(message_bytes).to_bytes(4, "big") + message_bytes
    bits = bytes_to_bits(payload)

    capacity = len(pixels) * 3

    if len(bits) > capacity:
        raise ValueError(
            f"Message is too large. Need {len(bits)} bits, "
            f"but the image can hold {capacity} bits."
        )

    modified_pixels = []
    bit_index = 0

    for pixel in pixels:
        channels = list(pixel)

        for channel_index in range(3):
            if bit_index < len(bits):
                channels[channel_index] = (
                    channels[channel_index] & 0xFE
                ) | bits[bit_index]

                bit_index += 1

        modified_pixels.append(tuple(channels))

    output = Image.new("RGB", image.size)
    output.putdata(modified_pixels)
    output.save(OUTPUT_IMAGE, format="PNG")

    print("LSB stego test image created successfully.")
    print(f"Source image : {SOURCE_IMAGE}")
    print(f"Output image : {OUTPUT_IMAGE}")
    print(f"Image size   : {image.size}")
    print(f"Payload size : {len(message_bytes)} bytes")
    print(f"Embedded bits: {len(bits)}")


if __name__ == "__main__":
    main()