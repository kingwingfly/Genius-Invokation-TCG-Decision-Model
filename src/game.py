from prelude import *
from player import Player
from char import Char

from enum import Enum
from random import randint


class State(Enum):
    Init = 0
    P1 = 1
    P2 = 2
    End = 3


class Game(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.players = [Player(1, ["qqman"]), Player(2, ["qqman"])]
        self.state = State.Init
        self.matrix = [
            [State.Init, State.Init, State.Init, State.Init],  # init
            [State.Init, State.P2, State.P1, State.End],  # turn
            [State.End, State.End, State.End, State.End],  # end
        ]

    def __str__(self) -> str:
        return f"State: {self.state}\n"

    @property
    def p1(self) -> Player:
        return self.players[0]

    @property
    def p2(self) -> Player:
        return self.players[1]

    def start(self):
        old = self.state
        assert old in [State.Init, State.End], f"Start at the wrong state {old}"
        self.state = State.P1 if randint(1, 2) == 1 else State.P2

    def turn(self):
        old = self.state
        now = self.matrix[1][old.value]
        self.state = now

    def end(self):
        old = self.state
        now = self.matrix[2][old.value]
        self.state = now

    def forward(self):
        while True:
            match self.state:
                case State.P1:
                    ...
                case State.P2:
                    ...
                case State.End:
                    break
                case State.Init:
                    self.start()


if __name__ == "__main__":
    game = Game()
    print(game)
    game.start()
    print(game)
    game.turn()
    print(game)
    game.turn()
    print(game)
    game.end()
    print(game)

    game()
