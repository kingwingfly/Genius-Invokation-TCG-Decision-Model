import json
import os


def get_attri(name: str) -> dict[str, int]:
    file_path = os.path.join(os.path.dirname(__file__), "attributions", "char.json")
    attribution = {}
    with open(file_path, 'r', encoding="utf-8") as f:
        attribution = json.load(f)
    return attribution[name]
