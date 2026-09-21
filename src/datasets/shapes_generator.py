""" Generates the synthetic 'shapes' dataset."""

import numpy as np
from PIL import Image, ImageDraw
import os
import json
import argparse
from scipy import ndimage

def check_overlap(mask: np.ndarray, new_region: np.ndarray, margin: int = 4) -> bool:
    kernel = np.ones((margin * 2 + 1, margin * 2 + 1))
    dilated = ndimage.binary_dilation(mask, structure=kernel)
    return bool(np.any(dilated & new_region))


def draw_circle(canvas: np.ndarray, cx: int, cy: int, r: int) -> np.ndarray:
    region = np.zeros_like(canvas)
    img = Image.fromarray(region.astype(np.uint8) * 255)
    draw = ImageDraw.Draw(img)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    return np.array(img) > 0


def draw_rectangle(canvas: np.ndarray, cx: int, cy: int, w: int, h: int) -> np.ndarray:
    region = np.zeros_like(canvas)
    img = Image.fromarray(region.astype(np.uint8) * 255)
    draw = ImageDraw.Draw(img)
    draw.rectangle([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2], fill=255)
    return np.array(img) > 0


def draw_triangle(canvas: np.ndarray, cx: int, cy: int, r: int) -> np.ndarray:
    region = np.zeros_like(canvas)
    img = Image.fromarray(region.astype(np.uint8) * 255)
    draw = ImageDraw.Draw(img)
    angle_offset = np.random.uniform(0, 2 * np.pi)
    pts = []
    for k in range(3):
        angle = angle_offset + k * (2 * np.pi / 3)
        pts.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))
    draw.polygon(pts, fill=255)
    return np.array(img) > 0

def generate_sample(
        c: int,
        image_size: int = 256,
        min_size: int = 18,
        max_size: int = 40,
        max_attempts: int = 200,
        shape_types: list = None
) -> tuple[np.ndarray, dict]:

    rng = np.random.default_rng()

    if shape_types is None:
        shape_types = ["circle", "rectangle", "triangle"]

    mask = np.zeros((image_size, image_size), dtype=bool)
    placed = []

    for _ in range(c):
        success = False
        for attempt in range(max_attempts):
            shape = rng.choice(shape_types)
            size = int(rng.integers(min_size, max_size + 1))
            margin = size + 4

            cx = int(rng.integers(margin, image_size - margin))
            cy = int(rng.integers(margin, image_size - margin))

            if shape == "circle":
                region = draw_circle(mask, cx, cy, size // 2)
            elif shape == "rectangle":
                w = size
                h = int(rng.integers(min_size, max_size + 1))
                region = draw_rectangle(mask, cx, cy, w, h)
            else:
                region = draw_triangle(mask, cx, cy, size // 2)

            if not check_overlap(mask, region, margin=4):
                mask |= region
                placed.append({"shape": shape, "cx": cx, "cy": cy, "size": size})
                success = True
                break

        if not success:
            raise RuntimeError(
                f"Failed to generate a valid mask without intersections for {max_attempts} attempts."
            )

    custom_struct = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], dtype=bool)
    _, con_comp = ndimage.label(mask, structure=custom_struct)
    assert con_comp == c, f"Error number of connected components does not match constraint."

    meta = {"constraint": c, "shapes": placed}
    return mask, meta

def generate_dataset(
        output_dir: str,
        samples_per_constraint: int = 50,
        c_range: tuple = (1, 10),
        image_size: int = 256
):
    """
    Entry point for generating the 'shapes' dataset.
    :param output_dir: Folder where the dataset will be stored.
    :param samples_per_constraint: Number of samples per constraint (desired number of objects present in each mask).
    :param c_range: Tuple (c_min, c_max) defining the instances of the zero betti numbers (number of connected components).
    :param image_size: Image resolution.
    :return: Meta data, that is also saved to output_dir/metadata.json
    """
    os.makedirs(output_dir, exist_ok=True)

    all_meta = []
    total = 0
    failed = 0

    c_min, c_max = c_range
    for c in range(c_min, c_max + 1):
        folder = os.path.join(output_dir, "masks", f"c_{c:02d}")
        subfolder = os.path.join("masks", f"c_{c:02d}")
        os.makedirs(folder, exist_ok=True)

        generated = 0
        attempts = 0

        while generated < samples_per_constraint:
            attempts += 1
            try:
                mask, meta = generate_sample(
                    c=c,
                    image_size=image_size
                )
                img_path = os.path.join(folder, f"sample_{generated:04d}.png")
                image_subpath = os.path.join(subfolder, f"sample_{generated:04d}.png")
                Image.fromarray(mask.astype(np.uint8) * 255).save(img_path)
                meta["path"] = image_subpath
                all_meta.append(meta)
                generated += 1
                total += 1
            except RuntimeError:
                failed += 1
                print("failed: ", failed)
                if failed > samples_per_constraint * 3:
                    print(f"  to many errors for c={c}, skipped.")
                    break

        print(f"  c={c:2d}: {generated}/{samples_per_constraint} samples generated")

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)

    print(f"\nfinished: save {total} samples to '{output_dir}'")
    print(f"meta data save to: {meta_path}")
    return all_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="shapes_dataset")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--c_min", type=int, default=1)
    parser.add_argument("--c_max", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating shapes dataset: c in [{args.c_min}, {args.c_max}], "
          f"{args.samples} Samples/c, {args.size}x{args.size}")

    generate_dataset(
        output_dir=args.output,
        samples_per_constraint=args.samples,
        c_range=(args.c_min, args.c_max),
        image_size=args.size
    )

