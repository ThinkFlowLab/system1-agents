"""Grid composition for the multi-snake recording."""

import math

from .ui import AMBER, CYAN, DIM, FG, GREEN, MUTED, RED, Canvas

TOP_BAR = 5


def cell_size(width, height):
    """Screen extent of one grid slot for a board of (width, height): the
    board, its borders, the side probability bars and one column of padding.
    The stock 24x16 board keeps the original 44x22 slot."""
    return max(34, width + 20), max(19, height + 6)


CELL_W, CELL_H = cell_size(24, 16)


def _draw_pending(c, ox, oy, g):
    bw, bh = g["width"], g["height"]
    c.put(oy, ox, f"#{g['seed']}", MUTED)
    c.put(oy, ox + 12, "pending", DIM)
    c.put(oy + 1, ox, "┌" + "─" * bw + "┐", DIM)
    c.put(oy + 2 + bh, ox, "└" + "─" * bw + "┘", DIM)
    for y in range(bh):
        c.put(oy + 2 + y, ox, "│", DIM)
        c.put(oy + 2 + y, ox + bw + 1, "│", DIM)


def _draw_cell(c, ox, oy, g):
    game = g["game"]
    decision = g.get("decision") or {}
    alive, won = game.get("alive", True), game.get("won", False)
    intervened = decision.get("intervened", False)
    border = RED if not alive else (GREEN if won else (AMBER if intervened else DIM))
    bw, bh = game["width"], game["height"]

    c.put(oy, ox, f"#{g['seed']} r{g.get('round', 1)}", MUTED)
    c.put(oy, ox + 12, f"S{game['score']}", FG)
    c.put(oy, ox + 18, f"{g.get('steps', 0)}st", MUTED)
    if intervened:
        c.put(oy, ox + 27, "SHIELD", AMBER)
    if not alive:
        c.put(oy, ox + 27, "DEAD", RED)
    if won:
        c.put(oy, ox + 27, "CLEAR", GREEN)

    c.put(oy + 1, ox, "┌" + "─" * bw + "┐", border)
    c.put(oy + 2 + bh, ox, "└" + "─" * bw + "┘", border)
    for y in range(bh):
        c.put(oy + 2 + y, ox, "│", border)
        c.put(oy + 2 + y, ox + bw + 1, "│", border)
        c.put(oy + 2 + y, ox + 1, "·" * bw, "#13272e")
    body = game["body"]
    for index, (x, y) in enumerate(body):
        fraction = 1 - index / max(1, len(body))
        color = (
            "#dcfff0"
            if index == 0
            else f"#{int(18 + 64 * fraction):02x}{int(73 + 150 * fraction):02x}{int(57 + 102 * fraction):02x}"
        )
        c.put(oy + 2 + y, ox + 1 + x, "█", color)
    if game.get("food") is not None:
        fx, fy = game["food"]
        c.put(oy + 2 + fy, ox + 1 + fx, "●", AMBER)

    probs = decision.get("probabilities") or {}
    executed = decision.get("executed", "—")
    rx = ox + bw + 3
    for index, direction in enumerate(("UP", "DOWN", "LEFT", "RIGHT")):
        p = probs.get(direction, 0)
        selected = direction == executed
        color = GREEN if selected else MUTED
        c.put(oy + 2 + index * 3, rx, direction[:2], color)
        c.put(oy + 2 + index * 3, rx + 3, "░" * 8, DIM)
        c.put(oy + 2 + index * 3, rx + 3, "█" * round(p * 8), color)
        c.put(oy + 2 + index * 3, rx + 12, f"{p:.2f}"[1:], color)
    risk = decision.get("dead_end_risk", 0)
    c.put(oy + 14, rx, "risk", MUTED)
    c.put(oy + 14, rx + 5, "░" * 6, DIM)
    c.put(oy + 14, rx + 5, "█" * round(risk * 6), AMBER if risk < 0.5 else RED)
    c.put(oy + 15, rx, "food", MUTED)
    fr = decision.get("food_reachable", 0)
    c.put(oy + 15, rx + 5, "░" * 6, DIM)
    c.put(oy + 15, rx + 5, "█" * round(fr * 6), CYAN)


def compose_multi(games, stats, cols=4, caption="SYSTEM1-OMNI"):
    """All slots are always drawn, so the canvas keeps one size for the whole
    recording; games without a frame yet render as pending placeholders."""
    n = len(games)
    rows = math.ceil(n / cols)
    boards = [g.get("game") or g for g in games] or [{}]
    cell_w, cell_h = cell_size(
        max(b.get("width", 24) for b in boards),
        max(b.get("height", 16) for b in boards),
    )
    width = 6 + cols * cell_w
    height = TOP_BAR + rows * cell_h + 2
    c = Canvas(width, height)
    c.put(1, 3, caption, MUTED)
    line = (
        f"{stats['alive']}/{n} alive · {stats['dps']:.0f} decisions/s · "
        f"{stats['mean_ms']:.1f} ms mean · score {stats['score']} · {stats['clock']}"
    )
    if stats.get("waiting"):
        line += f" · {stats['waiting']} waiting"
    c.put(2, 3, line, GREEN)
    c.put(3, 3, "─" * (width - 6), DIM)
    for i, g in enumerate(games):
        ox = 3 + (i % cols) * cell_w
        oy = TOP_BAR + (i // cols) * cell_h
        if g.get("pending"):
            _draw_pending(c, ox, oy, g)
        else:
            _draw_cell(c, ox, oy, g)
    return c
