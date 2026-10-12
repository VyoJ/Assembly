"""Fetch datasets + model weights into data/ so experiments never touch the network.

TODO(ML): implement each step. Layout this should produce:
    data/cmapss/{train,test,RUL}_FD001.txt
    data/mvtec/metal_nut/{train,test,ground_truth}/...
    data/weights/{yolo11n.pt, yolo11s.pt, sam3/, timm backbones...}
"""

from core.config import settings

STEPS = [
    ("C-MAPSS FD001", "NASA Prognostics Data Repository (or a Kaggle mirror) -> data/cmapss/"),
    ("MVTec AD metal_nut", "manual download after accepting the license at mvtec.com -> data/mvtec/metal_nut/"),
    ("YOLO weights", "ultralytics: YOLO('yolo11n.pt') downloads once; move into data/weights/"),
    ("SAM 3 weights", "Hugging Face (gated: request access first): huggingface_hub.snapshot_download -> data/weights/sam3/"),
    ("timm backbones", "timm.create_model(name, pretrained=True) once, save state_dict into data/weights/"),
]


def main() -> None:
    print(f"data dir: {settings.data_dir}")
    for name, how in STEPS:
        print(f"TODO  {name:20s} {how}")


if __name__ == "__main__":
    main()
