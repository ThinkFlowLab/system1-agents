# coding: utf-8
"""The ported Valen renderer: sizes, themes, colors and byte-for-byte determinism."""

import io
from unittest import TestCase, skipUnless

from s1a.agents._sokoban import parse_ascii, transition

try:
    from s1a.agents import _sokoban_render
except ImportError:  # the visual extra is not installed
    _sokoban_render = None

SIMPLE = "#####\n#@$.#\n#####"  # 5x3: walls around, player left of one box on a goal tile's row
ROOMY = "######\n#@$. #\n######"  # the same, one empty floor tile right of the goal
WALL = (0x47, 0x55, 0x69)  # classic #475569
FLOOR = (0xF8, 0xFA, 0xFC)  # classic #f8fafc
BOX = (0xD9, 0x9B, 0x45)  # classic #d99b45
PLAYER = (0x25, 0x63, 0xEB)  # classic #2563eb
GOAL = (0x0F, 0x76, 0x6E)  # classic #0f766e
WARM_WALL = (0x62, 0x55, 0x55)  # warm #625555


def png_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@skipUnless(_sokoban_render, "the visual extra (Pillow)")
class TestRender(TestCase):
    def render(self, text=SIMPLE, **kwargs):
        board, state = parse_ascii(text)
        return _sokoban_render.render(board, state, **kwargs)

    def test_size_is_tile_times_the_board(self):
        image = self.render(tile_size=24)
        self.assertEqual(image.size, (5 * 24, 3 * 24))

    def test_tile_colors(self):
        image = self.render(ROOMY, tile_size=24).convert("RGB")
        self.assertEqual(image.getpixel((0, 0)), WALL)  # a wall corner
        self.assertEqual(image.getpixel((4 * 24 + 12, 24 + 12)), FLOOR)  # the empty floor tile centre
        self.assertEqual(image.getpixel((2 * 24 + 9, 24 + 12)), BOX)  # the box fill, off its cross
        self.assertEqual(image.getpixel((24 + 12, 24 + 12)), PLAYER)  # the player circle
        self.assertEqual(image.getpixel((3 * 24 + 12, 24 + 12)), GOAL)  # the goal dot

    def test_warm_theme_walls(self):
        image = self.render(theme="warm", tile_size=24).convert("RGB")
        self.assertEqual(image.getpixel((0, 0)), WARM_WALL)

    def test_unknown_theme_or_tile_size_is_rejected(self):
        with self.assertRaises(ValueError):
            self.render(theme="neon")
        for size in (23, 129):
            with self.assertRaises(ValueError):
                self.render(tile_size=size)

    def test_the_same_state_renders_the_same_png_and_a_move_changes_it(self):
        board, state = parse_ascii(SIMPLE)
        before = png_bytes(_sokoban_render.render(board, state, tile_size=32))
        again = png_bytes(_sokoban_render.render(board, state, tile_size=32))
        moved, _, _ = transition(board, state, "right")
        after = png_bytes(_sokoban_render.render(board, moved, tile_size=32))
        self.assertEqual(before, again)
        self.assertNotEqual(before, after)
