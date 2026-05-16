#!/usr/bin/env python
"""Create MMDetection-format COCO open-world splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set


def parse_id_spec(spec: str) -> Set[int]:
    ids: Set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", maxsplit=1)
            ids.update(range(int(start), int(end) + 1))
        else:
            ids.add(int(chunk))
    return ids


def xywh_to_xyxy(box: Sequence[float]) -> List[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def build_category_maps(coco: dict, known_ids: Set[int]) -> tuple[list[str], Dict[int, int], Set[int]]:
    categories = sorted(coco["categories"], key=lambda item: item["id"])
    coco_ids = {item["id"] for item in categories}
    unknown_ids = coco_ids - known_ids
    known_categories = [item for item in categories if item["id"] in known_ids]
    known_names = [item["name"] for item in known_categories]
    known_id_to_label = {item["id"]: idx for idx, item in enumerate(known_categories)}
    return known_names, known_id_to_label, unknown_ids


def annotations_by_image(coco: dict) -> Dict[int, list[dict]]:
    grouped: Dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        grouped.setdefault(ann["image_id"], []).append(ann)
    return grouped


def convert_split(
    coco: dict,
    known_id_to_label: Dict[int, int],
    unknown_ids: Set[int],
    known_names: Sequence[str],
    image_prefix: str,
    mode: str,
    train_unknown_mode: str,
    keep_empty_images: bool,
) -> dict:
    if mode not in {"train", "val"}:
        raise ValueError(f"Unsupported mode: {mode}")

    unknown_label = len(known_names)
    grouped = annotations_by_image(coco)
    data_list = []

    for image in coco["images"]:
        instances = []
        for ann in grouped.get(image["id"], []):
            if ann.get("ignore", False):
                continue
            x1, y1, x2, y2 = xywh_to_xyxy(ann["bbox"])
            if ann.get("area", 0) <= 0 or x2 <= x1 or y2 <= y1:
                continue

            category_id = ann["category_id"]
            if category_id in known_id_to_label:
                instances.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "bbox_label": known_id_to_label[category_id],
                        "ignore_flag": 1 if ann.get("iscrowd", False) else 0,
                        "coco_category_id": category_id,
                    }
                )
            elif category_id in unknown_ids:
                if mode == "train" and train_unknown_mode == "ignore":
                    instances.append(
                        {
                            "bbox": [x1, y1, x2, y2],
                            "bbox_label": 0,
                            "ignore_flag": 1,
                            "coco_category_id": category_id,
                        }
                    )
                else:
                    instances.append(
                        {
                            "bbox": [x1, y1, x2, y2],
                            "bbox_label": unknown_label,
                            "ignore_flag": 1 if ann.get("iscrowd", False) else 0,
                            "coco_category_id": category_id,
                        }
                    )

        has_supervised_box = any(item["ignore_flag"] == 0 for item in instances)
        if keep_empty_images or has_supervised_box:
            data_list.append(
                {
                    "img_path": f"{image_prefix}/{image['file_name']}",
                    "img_id": image["id"],
                    "height": image["height"],
                    "width": image["width"],
                    "instances": instances,
                }
            )

    return {
        "metainfo": {
            "classes": [*known_names, "unknown"],
            "known_classes": list(known_names),
            "unknown_class": "unknown",
            "task": "coco_owod_t1",
        },
        "data_list": data_list,
    }


def write_metadata(
    out_dir: Path,
    known_names: Sequence[str],
    known_id_to_label: Dict[int, int],
    unknown_ids: Iterable[int],
) -> None:
    metadata = {
        "known_classes": list(known_names),
        "num_known_classes": len(known_names),
        "unknown_label": len(known_names),
        "known_coco_ids": sorted(known_id_to_label),
        "unknown_coco_ids": sorted(unknown_ids),
        "known_id_to_label": {str(k): v for k, v in known_id_to_label.items()},
    }
    dump_json(out_dir / "metadata.json", metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", required=True, type=Path)
    parser.add_argument("--val-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--known-ids", default="1-60")
    parser.add_argument("--train-prefix", default="train2017")
    parser.add_argument("--val-prefix", default="val2017")
    parser.add_argument(
        "--train-unknown-mode",
        choices=["unknown", "ignore"],
        default="unknown",
        help="Use `unknown` to train with COCO unknown boxes, or `ignore` for a stricter open-world split.",
    )
    parser.add_argument("--keep-empty-images", action="store_true")
    args = parser.parse_args()

    known_ids = parse_id_spec(args.known_ids)
    train_coco = load_json(args.train_json)
    val_coco = load_json(args.val_json)
    known_names, known_id_to_label, unknown_ids = build_category_maps(train_coco, known_ids)
    if not known_names:
        raise SystemExit("No known classes were selected. Check --known-ids.")

    train_split = convert_split(
        train_coco,
        known_id_to_label,
        unknown_ids,
        known_names,
        args.train_prefix,
        mode="train",
        train_unknown_mode=args.train_unknown_mode,
        keep_empty_images=args.keep_empty_images,
    )
    val_split = convert_split(
        val_coco,
        known_id_to_label,
        unknown_ids,
        known_names,
        args.val_prefix,
        mode="val",
        train_unknown_mode=args.train_unknown_mode,
        keep_empty_images=True,
    )

    dump_json(args.out_dir / "train_mmdet.json", train_split)
    dump_json(args.out_dir / "val_mmdet.json", val_split)
    write_metadata(args.out_dir, known_names, known_id_to_label, unknown_ids)
    print(f"Wrote {args.out_dir / 'train_mmdet.json'} with {len(train_split['data_list'])} images")
    print(f"Wrote {args.out_dir / 'val_mmdet.json'} with {len(val_split['data_list'])} images")
    print(f"Known classes: {len(known_names)}, unknown COCO ids: {len(unknown_ids)}")


if __name__ == "__main__":
    main()
