from typing import List, Tuple, Dict
from enum import Enum
import random
import os


class Player(Enum):
    X = "X"
    O = "O"


class BoardState(Enum):
    X_WON = "X_WON"
    O_WON = "O_WON"
    DRAW = "DRAW"
    IN_PROGRESS = "IN_PROGRESS"


class Board:
    _board: List[List[str]]

    def __init__(self):
        self._board = [[" " for _ in range(3)] for _ in range(3)]

    def mark(self, player: Player, row: int, col: int) -> None:
        if self._board[row][col] != " ":
            raise ValueError("Invalid move")
        self._board[row][col] = player.value

    def get_status(self) -> BoardState:
        for i in range(3):
            if all([spot == "X" for spot in self._board[i]]):
                return BoardState.X_WON
            if all([self._board[j][i] == "X" for j in range(3)]):
                return BoardState.X_WON

        if all([self._board[i][i] == "X" for i in range(3)]) or all(
            [self._board[i][2 - i] == "X" for i in range(3)]
        ):
            return BoardState.X_WON

        for i in range(3):
            if all([spot == "O" for spot in self._board[i]]):
                return BoardState.O_WON
            if all([self._board[j][i] == "O" for j in range(3)]):
                return BoardState.O_WON

        if all([self._board[i][i] == "O" for i in range(3)]) or all(
            [self._board[i][2 - i] == "O" for i in range(3)]
        ):
            return BoardState.O_WON

        if all([spot != " " for row in self._board for spot in row]):
            return BoardState.DRAW

        return BoardState.IN_PROGRESS

    def get_str_representation(self) -> str:
        return "\n---------\n".join([" | ".join(row) for row in self._board]) + "\n"

    def get_available_moves(self) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(3) for j in range(3) if self._board[i][j] == " "]

    def with_move(self, player: Player, row: int, col: int) -> "Board":
        new_board = Board()
        new_board._board = [row.copy() for row in self._board]
        new_board.mark(player, row, col)
        return new_board

    def get_key(self) -> Tuple[str]:
        return tuple([spot for row in self._board for spot in row])


class Controller:
    def get_move(self, board: Board) -> Tuple[int, int]:
        raise NotImplementedError

    should_print_board = False


class RandomController(Controller):
    def get_move(self, board: Board) -> Tuple[int, int]:
        return random.choice(board.get_available_moves())

    should_print_board = False


class CmdController(Controller):
    def get_move(self, board: Board) -> Tuple[int, int]:
        pos = input("Enter position (row * 3 + col): ")
        move = (int(pos) // 3, int(pos) % 3)
        if move not in board.get_available_moves():
            print("Invalid move")
            return self.get_move(board)
        return move

    should_print_board = True


class LearnedController(Controller):
    _value_dict = {}

    def value_fn(self, board: Board) -> float:
        return self._value_dict[board.get_key()]

    @staticmethod
    def init_value_dict() -> Dict[Board, float]:
        # if draw or lose, value is 0
        # if win, value is 1
        # else, value is 0.5

        d: Dict[Board, float] = {}

        def generate_all_boards(board: Board, player: Player) -> None:
            status = board.get_status()
            key = board.get_key()
            if status == BoardState.X_WON:
                d[key] = 1
                return
            if status == BoardState.O_WON or status == BoardState.DRAW:
                d[key] = 0
                return

            if key not in d:
                d[key] = 0.5

            for move in board.get_available_moves():
                new_board = board.with_move(player, move[0], move[1])
                generate_all_boards(
                    new_board, Player.X if player == Player.O else Player.O
                )

        empty_board = Board()
        generate_all_boards(empty_board, Player.X)
        return d

    def __init__(self, alpha=0.1, exploration_prob=0.2):
        self._value_dict = LearnedController.init_value_dict()
        self._alpha = alpha
        self._exploration_prob = exploration_prob

    _last_board = None

    def get_move(self, board: Board) -> Tuple[int, int]:
        if random.random() < self._exploration_prob:
            return random.choice(board.get_available_moves())

        self._last_board = board
        best_move = None
        best_value = -1
        for move in board.get_available_moves():
            new_board = board.with_move(Player.X, move[0], move[1])
            value = self.value_fn(new_board)
            if value > best_value:
                best_move = move
                best_value = value

        # learn if last board is not None
        if self._last_board is not None:
            last = self._value_dict[self._last_board.get_key()]

            self._value_dict[self._last_board.get_key()] = last + self._alpha * (
                best_value - last
            )
        return best_move

    def save_value_dict(self, filename: str = "data/value_dict.txt"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            for board, value in self._value_dict.items():
                f.write(f"{board}---{value}\n")

    def try_load_value_dict(self, filename: str = "data/value_dict.txt"):
        try:
            with open(filename, "r") as f:
                for line in f:
                    board_str, value_str = line.split("---")
                    key = tuple(board_str[1:-1].split(", "))
                    value = float(value_str)
                    self._value_dict[key] = value

        except FileNotFoundError:
            pass


class Game:
    def __init__(self, controller1: Controller, controller2: Controller):
        self._board = Board()
        self._controller1 = controller1
        self._controller2 = controller2

    def play(self):
        while True:
            if self._controller1.should_print_board:
                print(self._board.get_str_representation())
            move = self._controller1.get_move(self._board)
            self._board.mark(Player.X, move[0], move[1])
            if self._board.get_status() != BoardState.IN_PROGRESS:
                break

            if self._controller2.should_print_board:
                print(self._board.get_str_representation())
            move = self._controller2.get_move(self._board)
            self._board.mark(Player.O, move[0], move[1])
            if self._board.get_status() != BoardState.IN_PROGRESS:
                break

        print(self._board.get_str_representation())
        status = self._board.get_status()
        if status == BoardState.X_WON:
            print("X won")
        elif status == BoardState.O_WON:
            print("O won")
        else:
            print("Draw")


class Trainer:
    def __init__(self):
        self._learned = LearnedController()
        self._random = RandomController()

    def train(self, num_games: int):
        for i in range(num_games):
            game = Game(self._learned, self._random)
            game.play()
            print("Game completed: ", i)
            self._learned.save_value_dict()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="play")

    args = parser.parse_args()

    if args.mode == "play":
        lc = LearnedController()
        lc.try_load_value_dict()
        cmd_controller = CmdController()
        while True:
            game = Game(lc, cmd_controller)
            game.play()

    if args.mode == "train":
        trainer = Trainer()
        trainer.train(1000)
