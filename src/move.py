# move.py

class Move:
    __slots__ = ("from_pos", "to_pos", "captured")

    def __init__(self, from_pos, to_pos, captured=0):
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.captured = captured

    def __repr__(self):
        return f"Move({self.from_pos} -> {self.to_pos}, cap={self.captured})"