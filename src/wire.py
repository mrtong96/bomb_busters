# class to handle some wire logic
from typing import Optional

BLUE = 0
YELLOW = 1
RED = 5

class Wire:
    def __init__(self, raw_int: Optional[int] = None, rank: int=0, color='blue'):
        """

        Args:
            raw_int: integer representing the rank of the wire, can be optional
            rank: 0 represents unspecified, otherwise can be anywhere in the range of 1-12 for blue,
                1-11 for yellow and red
            color: red/yellow/blue
        """
        if raw_int is None:
            self.rank = rank
            if isinstance(color, str):
                color = {'blue': BLUE, 'yellow': YELLOW, 'red': RED}[color.lower()]
            self.color = color
        else:
            self.rank = raw_int // 10
            self.color = raw_int % 10

            if self.color not in {BLUE, YELLOW, RED}:
                raise RuntimeError(f"wrong ones digit: {raw_int}")

        self.raw_int = 10 * self.rank + self.color

    def __eq__(self, other):
        return self.raw_int == other.raw_int

    def __lt__(self, other):
        return self.raw_int < other.raw_int

    def __hash__(self):
        return hash(self.raw_int)

    def __repr__(self):
        return self.raw_int.__repr__()
