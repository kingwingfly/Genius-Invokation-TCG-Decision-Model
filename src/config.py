from torch.cuda import is_available as is_cuda_available
from torch.backends.mps import is_available as is_mps_available

import json
import os

DEVICE = "cuda" if is_cuda_available() else "mps" if is_mps_available() else "cpu"

file_path = os.path.join(os.path.dirname(__file__), "attributions", "char.json")
CHAR_ATTRI = {}
with open(file_path, "r", encoding="utf-8") as f:
    CHAR_ATTRI = json.load(f)


# todo Test 冻结下的反应
# ER = Elemental Reactions
ER_MATRIX = [
    # [冻,    无(物),   雷,     草,     火,      水,     冰]
    [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],  # None
    [(1, 2), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)],  # 无(物)
    [(0, 0), (2, 0), (2, 0), (1, 1), (1, 2), (1, 1), (1, 1)],  # 雷
    [(0, 0), (3, 0), (1, 1), (3, 0), (1, 1), (1, 1), (6, 0)],  # 草
    [(1, 2), (4, 0), (1, 2), (1, 1), (4, 0), (1, 2), (1, 2)],  # 火
    [(0, 0), (5, 0), (1, 1), (1, 1), (1, 2), (5, 0), (0, 1)],  # 水
    [(0, 0), (6, 0), (1, 1), (3, 0), (1, 2), (0, 1), (6, 0)],  # 冰
    [(0, 0), (1, 0), (1, 0), (3, 0), (1, 0), (1, 0), (1, 0)],  # 风
    [(0, 0), (7, 0), (1, 1), (3, 0), (1, 1), (1, 1), (1, 1)],  # 岩
]

ELEM_DICT = {
    0: "穿",
    1: "无",
    2: "雷",
    3: "草",
    4: "火",
    5: "水",
    6: "冰",
    7: "风",
    8: "岩",
}


def ER_FSM_checker():
    for i in range(7):
        for j in range(8):
            new, addition = ER_MATRIX[j][i]
            print(
                f"get {ELEM_DICT[j]}; rom {ELEM_DICT[i]} to {ELEM_DICT[new]}; addtion {addition}"
            )


if __name__ == "__main__":
    print(f"DEVICE: {DEVICE}")
    ER_FSM_checker()
