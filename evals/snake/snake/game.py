"""Deterministic Snake rules and a separately identified cycle safety planner."""

import random
from collections import deque
from dataclasses import dataclass

DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
VECTORS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}


def hamiltonian_cycle(width, height):
    """Visit each square once with adjacent steps, including the closing edge."""
    if min(width, height) < 4 or (width % 2 and height % 2):
        raise ValueError("Board dimensions must be >= 4, with at least one even dimension")
    if height % 2:
        return [(y, x) for x, y in hamiltonian_cycle(height, width)]
    path = [(0, 0)]
    for y in range(height):
        xs = range(1, width) if y % 2 == 0 else range(width - 1, 0, -1)
        path.extend((x, y) for x in xs)
    path.extend((0, y) for y in range(height - 1, 0, -1))
    return path


@dataclass(frozen=True)
class MoveInfo:
    direction: str
    legal: bool
    safe: bool
    advance: int
    reason: str
    eats: bool


class SnakeGame:
    def __init__(self, width=24, height=16, seed=7, initial_length=6):
        self.width, self.height, self.seed = width, height, seed
        self.cycle = hamiltonian_cycle(width, height)
        self.indices = {cell: index for index, cell in enumerate(self.cycle)}
        self.capacity = width * height
        if not 2 <= initial_length < self.capacity:
            raise ValueError("Initial length must be >= 2 and smaller than the board")
        self.initial_length = initial_length
        self.rng = random.Random(seed)
        start = self.indices[(width // 2, height // 2)]
        self.body = deque(self.cycle[(start - i) % self.capacity] for i in range(initial_length))
        self.score = self.ticks = 0
        self.alive, self.won = True, False
        self.death_reason = None
        self.food = self._spawn_food()

    @property
    def head(self):
        return self.body[0]

    def _spawn_food(self):
        occupied = set(self.body)
        empty = [cell for cell in self.cycle if cell not in occupied]
        return self.rng.choice(empty) if empty else None

    def target(self, direction):
        dx, dy = VECTORS[direction]
        return self.head[0] + dx, self.head[1] + dy

    def legal_reason(self, direction):
        x, y = cell = self.target(direction)
        if not (0 <= x < self.width and 0 <= y < self.height):
            return "wall"
        if cell == self.body[1]:
            return "reverse"
        occupied = set(self.body)
        if cell != self.food:
            occupied.remove(self.body[-1])  # The tail moves on a non-growing step.
        return "body" if cell in occupied else "legal"

    def moves(self):
        if not self.alive or self.won:
            return []
        head_index = self.indices[self.head]
        tail_distance = (self.indices[self.body[-1]] - head_index) % self.capacity
        food_distance = (self.indices[self.food] - head_index) % self.capacity
        moves = []
        for direction in DIRECTIONS:
            reason = self.legal_reason(direction)
            legal = reason == "legal"
            target = self.target(direction)
            advance = (self.indices.get(target, head_index) - head_index) % self.capacity
            eats = target == self.food
            safe = legal
            if safe and (advance > tail_distance or (advance == tail_distance and eats)):
                safe, reason = False, "would cross the tail"
            if safe and (advance == 0 or advance > food_distance):
                safe, reason = False, "would skip the food on the safe route"
            moves.append(MoveInfo(direction, legal, safe, advance, reason, eats))
        return moves

    def food_reachability(self):
        """Current empty-cell connectivity; the occupied tail is not treated as empty."""
        blocked = set(self.body) - {self.head}
        visited = {self.head}
        queue = deque([self.head])
        while queue:
            x, y = queue.popleft()
            for dx, dy in VECTORS.values():
                cell = x + dx, y + dy
                if (
                    0 <= cell[0] < self.width
                    and 0 <= cell[1] < self.height
                    and cell not in blocked
                    and cell not in visited
                ):
                    visited.add(cell)
                    queue.append(cell)
        return self.food in visited, len(visited)

    def step(self, direction):
        if not self.alive or self.won:
            raise RuntimeError("Cannot step a finished game")
        if direction not in DIRECTIONS:
            raise ValueError(f"Unknown direction: {direction}")
        self.ticks += 1
        reason = self.legal_reason(direction)
        if reason != "legal":
            self.alive, self.death_reason = False, reason
            return False
        target = self.target(direction)
        self.body.appendleft(target)
        if target == self.food:
            self.score += 1
            if len(self.body) == self.capacity:
                self.won, self.food = True, None
            else:
                self.food = self._spawn_food()
            return True
        self.body.pop()
        return False

    def cycle_order_valid(self):
        indices = [self.indices[cell] for cell in reversed(self.body)]
        distances = [(b - a) % self.capacity for a, b in zip(indices, indices[1:])]
        return all(d > 0 for d in distances) and sum(distances) < self.capacity

    def snapshot(self):
        return {
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "body": [list(cell) for cell in self.body],
            "food": list(self.food) if self.food else None,
            "score": self.score,
            "length": len(self.body),
            "ticks": self.ticks,
            "alive": self.alive,
            "won": self.won,
            "death_reason": self.death_reason,
        }
