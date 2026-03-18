import os
import json
import time
from pathlib import Path
import cv2

from config import INPUT_DIR, OUTPUT_DIR, JSON_DIR, RENDER_DIR, load_craft
from pipeline import process_image
from chat_renderer import render_chat

def main():
    pipeline_start = time.time()
    print("\n[PIPELINE] Starting Messenger OCR pipeline\n")

    craft_net = load_craft()

    images = list(Path(INPUT_DIR).glob("*"))
    print(f"[INPUT] Found {len(images)} images")

    total_images = len(images)

    for index, path in enumerate(images, start=1):
        image_start = time.time()
        print(
            f"\n[PIPELINE] Image {index}/{total_images}: "
            f"{Path(path).name}"
        )
        overlay, meta = process_image(str(path), craft_net)

        out = os.path.join(OUTPUT_DIR, path.name)
        cv2.imwrite(out, overlay)

        json_path = os.path.join(
            JSON_DIR,
            Path(path.name).stem + ".json"
        )

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        rendered_chat = render_chat(meta)
        render_path = os.path.join(
            RENDER_DIR,
            Path(path.name).stem + "_chat.png"
        )
        cv2.imwrite(render_path, rendered_chat)

        image_runtime = time.time() - image_start
        print(
            f"[OUTPUT] Saved results for {Path(path).name} "
            f"in {image_runtime:.2f}s"
        )
        print(f"[OUTPUT] Saved rendered chat to {render_path}")

    total_runtime = time.time() - pipeline_start
    print(
        f"\n[PIPELINE] Finished processing all images "
        f"in {total_runtime:.2f}s"
    )

if __name__ == "__main__":
    main()