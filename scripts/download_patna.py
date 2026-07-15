import os
import sys
import json
import yaml
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def search_planetary_computer(
    bbox: list[float],
    date_range: str,
    max_cloud_cover: int = 10,
    limit: int = 10,
):
    try:
        import pystac_client
        import planetary_computer
    except ImportError:
        print("Install pystac-client and planetary-computer:")
        print("  pip install pystac-client planetary-computer")
        return []

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        limit=limit,
        sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
    )

    items = list(search.items())
    print(f"Found {len(items)} scenes")
    return items


def download_item(item, output_dir: str, bands: list[str] | None = None):
    try:
        import planetary_computer
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError:
        print("Install required packages: pip install planetary-computer rasterio")
        return

    os.makedirs(output_dir, exist_ok=True)

    date = item.datetime.strftime("%Y%m%d")
    item_dir = os.path.join(output_dir, f"S2_{date}")
    os.makedirs(item_dir, exist_ok=True)

    for asset_key, asset in item.assets.items():
        if bands and asset_key not in bands:
            continue

        href = asset.href
        if not href:
            continue

        print(f"  Downloading {asset_key}...")
        try:
            with rasterio.open(href) as src:
                data = src.read()
                meta = src.meta.copy()

            out_path = os.path.join(item_dir, f"{asset_key}.tif")
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(data)
        except Exception as e:
            print(f"  Failed to download {asset_key}: {e}")


def download_patna_imagery(
    area_name: str = "full",
    t1_date: str = "2022-01-01/2022-12-31",
    t2_date: str = "2024-01-01/2024-12-31",
    max_cloud_cover: int = 10,
):
    from scripts.prepare_patna_data import PATNA_AOI

    if area_name not in PATNA_AOI:
        print(f"Unknown area: {area_name}")
        print(f"Available areas: {list(PATNA_AOI.keys())}")
        return

    area = PATNA_AOI[area_name]
    bbox = area["bounds"]

    print(f"Area: {area['name']}")
    print(f"BBox: {bbox}")
    print()

    output_base = os.path.join("data", "patna", area_name)

    print(f"Searching T1 scenes ({t1_date})...")
    t1_items = search_planetary_computer(bbox, t1_date, max_cloud_cover, limit=5)
    if t1_items:
        best_t1 = t1_items[0]
        print(f"Best T1: {best_t1.datetime.strftime('%Y-%m-%d')} (cloud: {best_t1.properties.get('eo:cloud_cover', 'N/A')}%)")
        t1_dir = os.path.join(output_base, "t1_raw")
        download_item(best_t1, t1_dir, bands=["B02", "B03", "B04", "B08", "SCL"])
    else:
        print("No T1 scenes found")

    print(f"\nSearching T2 scenes ({t2_date})...")
    t2_items = search_planetary_computer(bbox, t2_date, max_cloud_cover, limit=5)
    if t2_items:
        best_t2 = t2_items[0]
        print(f"Best T2: {best_t2.datetime.strftime('%Y-%m-%d')} (cloud: {best_t2.properties.get('eo:cloud_cover', 'N/A')}%)")
        t2_dir = os.path.join(output_base, "t2_raw")
        download_item(best_t2, t2_dir, bands=["B02", "B03", "B04", "B08", "SCL"])
    else:
        print("No T2 scenes found")

    print("\nDownload complete.")
    print(f"Data saved to: {output_base}")
    print("Next: Run prepare_patna_data.py to tile images into patches")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download Sentinel-2 imagery for Patna AOI")
    parser.add_argument("--area", type=str, default="full", choices=["boring_road", "kurji", "full"])
    parser.add_argument("--t1-date", type=str, default="2022-01-01/2022-12-31")
    parser.add_argument("--t2-date", type=str, default="2024-01-01/2024-12-31")
    parser.add_argument("--max-cloud", type=int, default=10)
    args = parser.parse_args()

    download_patna_imagery(
        area_name=args.area,
        t1_date=args.t1_date,
        t2_date=args.t2_date,
        max_cloud_cover=args.max_cloud,
    )
