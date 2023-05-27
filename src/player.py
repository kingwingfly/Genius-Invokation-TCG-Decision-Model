from prelude import *
from char import Char


class Player(nn.Module):
    def __init__(self, id: int, names: list[str]) -> None:
        super().__init__()
        self.__id = id
        self.chars = [Char(name) for name in names]

    def __str__(self) -> str:
        return f"Player id: {self.__id}"


if __name__ == "__main__":
    ...
