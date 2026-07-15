import os
import sys
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.transform import from_bounds
except ImportError:
    rasterio = None


PATNA_AOI = {
    "boring_road": {
        "name": "Boring Road, Patna",
        "bounds": [85.095, 25.605, 85.115, 25.618],
        "description": "Commercial corridor with dense urban development",
    },
    "kurji": {
        "name": "Kurji, Patna",
        "bounds": [85.080, 25.578, 85.098, 25.592],
        "description": "Residential expansion and institutional area",
    },
    "full": {
        "name": "Patna - Full AOI",
        "bounds": [85.08, 25.57, 85.12, 25.62],
        "description": "Combined Boring Road + Kurji area",
    },
}


def tile_image(
    image_path: str,
    output_dir: str,
    patch_size: int = 256,
    overlap: int = 32,
):
    if rasterio is None:
        raise ImportError("rasterio is required for tiling. Install with: pip install rasterio")

    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(image_path) as src:
        meta = src.meta.copy()
        width = src.width
        height = src.height

        stride = patch_size - overlap
        tiles_created = 0

        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                window = Window(x, y, patch_size, patch_size)
                tile = src.read(window=window)

                tile_meta = meta.copy()
                tile_meta.update({
                    "width": patch_size,
                    "height": patch_size,
                    "transform": rasterio.windows.transform(window, src.transform),
                })

                tile_path = os.path.join(output_dir, f"tile_{y:04d}_{x:04d}.tif")
                with rasterio.open(tile_path, "w", **tile_meta) as dst:
                    dst.write(tile)
                tiles_created += 1

        print(f"Created {tiles_created} tiles from {image_path}")
        return tiles_created


def compute_ndvi_from_tif(nir_path: str, red_path: str, output_path: str):
    if rasterio is None:
        raise ImportError("rasterio is required. Install with: pip install rasterio")

    with rasterio.open(nir_path) as nir_src:
        nir = nir_src.read(1).astype(np.float32)
        meta = nir_src.meta.copy()

    with rasterio.open(red_path) as red_src:
        red = red_src.read(1).astype(np.float32)

    ndvi = (nir - red) / (nir + red + 1e-10)
    ndvi = np.clip(ndvi, -1, 1)

    meta.update({"count": 1, "dtype": "float32", "nodata": -9999})
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(ndvi.astype(np.float32), 1)

    print(f"NDVI saved to {output_path}")


def prepare_patna_data(config_path: str = "configs/inference.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    aoi = config["patna_aoi"]
    patch_size = config["inference"]["input_size"]
    overlap = config["inference"]["overlap"]

    output_base = os.path.join("data", "patna")
    os.makedirs(output_base, exist_ok=True)

    print(f"Patna AOI: {aoi['name']}")
    print(f"Bounds: {aoi['bounds']}")
    print(f"Date range: {aoi['t1_date']} to {aoi['t2_date']}")
    print(f"Max cloud cover: {aoi['max_cloud_cover']}%")
    print()

    for area_name, area_info in PATNA_AOI.items():
        area_dir = os.path.join(output_base, area_name)
        t1_dir = os.path.join(area_dir, "t1")
        t2_dir = os.path.join(area_dir, "t2")
        os.makedirs(t1_dir, exist_ok=True)
        os.makedirs(t2_dir, exist_ok=True)

        print(f"Area: {area_info['name']}")
        print(f"  Bounds: {area_info['bounds']}")
        print(f"  Output: {area_dir}")

        t1_files = list(Path(t1_dir).glob("*.tif"))
        t2_files = list(Path(t2_dir).glob("*.tif"))

        if t1_files and t2_files:
            print(f"  T1 tiles: {len(t1_files)}")
            print(f"  T2 tiles: {len(t2_files)}")
        else:
            print("  No data found - download Sentinel-2 data first")
        print()

    print("Data preparation complete.")
    print("Next: Use scripts/download_patna.py to fetch Sentinel-2 imagery")


if __name__ == "__main__":
    prepare_patna_data()
