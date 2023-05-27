from __future__ import annotations
from prelude import *
import helper

# todo Test 冻结下的反应
MATRIX = [
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


class BasicChar:
    def __init__(self, attri: dict[str, int]) -> None:
        # todo may use tensor
        self._upper_hp = attri["hp"]
        self._hp = self._upper_hp
        self._uppper_mp = attri["mp"]
        self._mp = 0
        self._shield = attri["shield"]  # initial shield
        self.state = 1  # 1 for alive; 0 for dead
        self.teammates = []
        self.attach = 1  # this need a FSM
        self.a_effect = attri["a"]
        self.e_effect = attri["e"]
        self.q_effect = attri["q"]

    @property
    def hp(self):
        assert (
            self._hp <= self._upper_hp
        ), f"illegal hp number {self._hp}, which should <= {self._upper_hp}"
        return self._hp + self._shield

    def has_shield(self) -> bool:
        return bool(self._shield)

    def can_q(self) -> bool:
        return self._mp == self._uppper_mp

    def injury(self, injury: int, elem: int):
        assert self.state, "attacking a dead char"
        old = self.attach
        self.attach, addition = MATRIX[elem][self.attach] if elem else (self.attach, 0)
        for _ in range(injury + addition):
            if self.has_shield():
                self._shield -= 1
                continue
            self._hp -= 1
            if not self._hp:
                self.state = 0
                break
        self.additional(elem, old)

    def cure(self, num: int):
        if self.state:
            self._hp = min(self._hp + num, self._upper_hp)

    def shield(self, num: int):
        if self.state:
            self._shield += num

    def charge(self):
        if self.state:
            self._mp = min(self._mp + 1, self._uppper_mp)

    def relive(self):
        self.state = 1
        self.cure(1)

    def additional(self, elem: int, old: int):
        # 扩散
        if elem == 7 and old in [2, 4, 5, 6]:
            list(map(lambda x: x.injury(1, old), self.teammates))
        # 超导
        if (elem, old) in [(2, 6), (6, 2)]:
            list(map(lambda x: x.injury(1, 0), self.teammates))

    def a(self, target: Char):
        if not (self.attach and target.state):
            return
        # todo consider buff, weapon, etc
        injury = self.a_effect["damage"]
        elem = self.a_effect["elem"]
        target.injury(injury, elem)
        self.charge()

    def e(self, target: Char):
        if not (self.attach and target.state):
            return
        # todo consider buff, weapon, etc
        injury = self.e_effect["damage"]
        elem = self.e_effect["elem"]
        target.injury(injury, elem)
        self.charge()

    def q(self, target: Char):
        if not (self.attach and target.state):
            return
        if not self.can_q():
            return
        # todo consider buff, weapon, etc
        injury = self.q_effect["damage"]
        elem = self.q_effect["elem"]
        target.injury(injury, elem)
        self._mp = 0


class Char(BasicChar):
    def __init__(self, name: str) -> None:
        super().__init__(helper.get_attri(name))
        self.name = name

    def __str__(self) -> str:
        return f"Name: {self.name}\nState: hp: {self._hp}\t mp: {self._mp}\t shield: {self._shield}\t attach: {ELEM_DICT[self.attach]}\n"


def create_chars(names: list[str]) -> list[Char]:
    assert len(names) == 6, f"expect 6 names, got {len(names)} only"
    chars = [Char(name) for name in names]
    for i in range(6):
        if i <= 2:
            lst = [0, 1, 2]
            lst.remove(i)
            chars[i].teammates.extend([chars[j] for j in lst])
        if i > 2:
            lst = [3, 4, 5]
            lst.remove(i)
            chars[i].teammates.extend([chars[j] for j in lst])
    return chars


def FSM_checker():
    for i in range(7):
        for j in range(8):
            new, addition = MATRIX[j][i]
            print(
                f"get {ELEM_DICT[j]}; rom {ELEM_DICT[i]} to {ELEM_DICT[new]}; addtion {addition}"
            )


if __name__ == "__main__":
    char1, char2, char3, char4, char5, char6 = create_chars(
        ["qqman_fire", "qqman_wind"] * 3
    )
    # injury, cure, relive, shield tests
    char1.injury(10, 1)
    print(char1)
    char1.cure(10)
    print(char1)
    char1.relive()
    print(char1)
    char1.cure(10)
    print(char1)
    char1.shield(3)
    print(char1)
    # a, e, q tests
    char1.a(char4)
    char4.a(char1)
    char1.e(char4)
    print(char1)
    print(char4)
    char2.e(char4)
    print(char1)
    print(char2)
    print(char4)
    print(char5)
    print(char6)
