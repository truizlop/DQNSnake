"""Live activation dashboard for Swift DQN play mode.

Reads line-delimited JSON snapshots from stdin and renders a pygame window.
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any

try:
    import pygame
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("pygame is required for activation_viewer.py") from exc


BACKGROUND = (18, 20, 24)
PANEL = (30, 34, 42)
BORDER = (78, 88, 105)
TEXT = (232, 236, 243)
MUTED = (154, 164, 180)
SELECTED = (255, 194, 87)
BAR = (66, 153, 225)
NEGATIVE_BAR = (229, 92, 92)
ACTIONS = ["left", "up", "right", "down"]


def heat_color(value: int) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, value / 255.0))
    # Compact viridis-like ramp: dark purple -> blue -> green -> yellow.
    stops = [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ]
    position = t * (len(stops) - 1)
    index = min(len(stops) - 2, int(position))
    local = position - index
    a = stops[index]
    b = stops[index + 1]
    return tuple(int(a[i] + (b[i] - a[i]) * local) for i in range(3))


def make_heatmap_surface(image: dict[str, Any]) -> pygame.Surface:
    width = int(image["width"])
    height = int(image["height"])
    pixels = image["pixels"]
    rgb = bytearray(width * height * 3)
    for offset, value in enumerate(pixels):
        r, g, b = heat_color(int(value))
        base = offset * 3
        rgb[base] = r
        rgb[base + 1] = g
        rgb[base + 2] = b
    return pygame.image.frombuffer(bytes(rgb), (width, height), "RGB").copy()


def fit_rect(source_size: tuple[int, int], target: pygame.Rect) -> pygame.Rect:
    source_w, source_h = source_size
    scale = min(target.width / source_w, target.height / source_h)
    width = max(1, int(source_w * scale))
    height = max(1, int(source_h * scale))
    return pygame.Rect(
        target.x + (target.width - width) // 2,
        target.y + (target.height - height) // 2,
        width,
        height,
    )


def draw_text(surface: pygame.Surface, font: pygame.font.Font, text: str, pos: tuple[int, int], color=TEXT) -> None:
    surface.blit(font.render(text, True, color), pos)


def draw_panel(
    screen: pygame.Surface,
    font: pygame.font.Font,
    title: str,
    image: dict[str, Any],
    rect: pygame.Rect,
    small_font: pygame.font.Font | None = None,
) -> None:
    pygame.draw.rect(screen, PANEL, rect, border_radius=10)
    pygame.draw.rect(screen, BORDER, rect, width=1, border_radius=10)
    draw_text(screen, font, title, (rect.x + 12, rect.y + 10))
    heatmap = make_heatmap_surface(image)
    content = pygame.Rect(rect.x + 12, rect.y + 38, rect.width - 24, rect.height - 50)
    fitted = fit_rect(heatmap.get_size(), content)
    scaled = pygame.transform.scale(heatmap, fitted.size)
    screen.blit(scaled, fitted)
    if small_font is not None:
        draw_tile_labels(screen, small_font, image, fitted)


def draw_tile_labels(
    screen: pygame.Surface,
    font: pygame.font.Font,
    image: dict[str, Any],
    fitted: pygame.Rect,
) -> None:
    labels = image.get("tile_labels") or []
    tile_width = image.get("tile_width")
    tile_height = image.get("tile_height")
    tile_columns = image.get("tile_columns")
    tile_padding = image.get("tile_padding", 0)
    if not labels or not tile_width or not tile_height or not tile_columns:
        return
    if len(labels) > 16:
        return

    scale_x = fitted.width / int(image["width"])
    scale_y = fitted.height / int(image["height"])
    for index, label in enumerate(labels):
        column = index % int(tile_columns)
        row = index // int(tile_columns)
        raw_x = column * (int(tile_width) + int(tile_padding))
        raw_y = row * (int(tile_height) + int(tile_padding))
        x = fitted.x + int(raw_x * scale_x) + 4
        y = fitted.y + int(raw_y * scale_y) + 4
        text = font.render(str(label), True, TEXT)
        background = pygame.Rect(x - 2, y - 1, text.get_width() + 4, text.get_height() + 2)
        pygame.draw.rect(screen, (0, 0, 0), background, border_radius=3)
        screen.blit(text, (x, y))


def draw_q_values(
    screen: pygame.Surface,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    q_values: list[float],
    selected_index: int,
    rect: pygame.Rect,
) -> None:
    pygame.draw.rect(screen, PANEL, rect, border_radius=10)
    pygame.draw.rect(screen, BORDER, rect, width=1, border_radius=10)
    draw_text(screen, font, "Q-values", (rect.x + 12, rect.y + 10))

    max_abs = max(1e-6, max(abs(v) for v in q_values))
    bar_left = rect.x + 84
    bar_width = rect.width - 130
    baseline = bar_left + bar_width // 2
    y = rect.y + 48
    for index, value in enumerate(q_values):
        color = SELECTED if index == selected_index else TEXT
        draw_text(screen, small_font, ACTIONS[index], (rect.x + 12, y), color)
        draw_text(screen, small_font, f"{value: .3f}", (rect.right - 62, y), color)
        magnitude = int((abs(value) / max_abs) * (bar_width // 2))
        if value >= 0:
            bar_rect = pygame.Rect(baseline, y + 3, magnitude, 12)
            pygame.draw.rect(screen, BAR, bar_rect, border_radius=4)
        else:
            bar_rect = pygame.Rect(baseline - magnitude, y + 3, magnitude, 12)
            pygame.draw.rect(screen, NEGATIVE_BAR, bar_rect, border_radius=4)
        pygame.draw.line(screen, MUTED, (baseline, y), (baseline, y + 18), width=1)
        y += 34


def draw_dashboard(screen: pygame.Surface, snapshot: dict[str, Any], fonts: tuple[pygame.font.Font, pygame.font.Font]) -> None:
    font, small_font = fonts
    screen.fill(BACKGROUND)
    episode = snapshot["episode"]
    step = snapshot["step"]
    action = snapshot["action"]
    q_values = [float(value) for value in snapshot["q_values"]]
    selected = int(snapshot["selected_action_index"])
    images = snapshot["images"]

    draw_text(screen, font, f"Snake DQN activation dashboard", (24, 18))
    draw_text(screen, small_font, f"episode {episode} / step {step} / selected action: {action}", (24, 46), MUTED)

    draw_panel(
        screen,
        font,
        "Input stack: 4 binary frames, display-aligned",
        images["input"],
        pygame.Rect(24, 76, 360, 360),
        small_font,
    )
    draw_panel(
        screen,
        font,
        "conv1: 16 channels, display-aligned",
        images["conv1"],
        pygame.Rect(408, 76, 360, 360),
        small_font,
    )
    draw_panel(
        screen,
        font,
        "conv2: 32 channels, display-aligned",
        images["conv2"],
        pygame.Rect(792, 76, 360, 360),
        None,
    )
    draw_panel(screen, font, "Dense layer: 256 units", images["dense"], pygame.Rect(24, 462, 552, 220))
    draw_panel(screen, font, "Q-value heatmap", images["qvalues"], pygame.Rect(600, 462, 180, 220))
    draw_q_values(screen, font, small_font, q_values, selected, pygame.Rect(804, 462, 348, 220))

    draw_text(
        screen,
        small_font,
        "Spatial heatmaps are transposed for display to match pygame orientation. Values are normalized per image/channel for readability.",
        (24, 704),
        MUTED,
    )
    pygame.display.flip()


def main() -> int:
    pygame.init()
    pygame.display.set_caption("Snake DQN Activations")
    screen = pygame.display.set_mode((1176, 744))
    fonts = (pygame.font.SysFont("Menlo", 18) or pygame.font.Font(None, 18), pygame.font.SysFont("Menlo", 14) or pygame.font.Font(None, 14))

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 0
        snapshot = json.loads(raw)
        draw_dashboard(screen, snapshot, fonts)

    # Keep the final frame visible briefly if the publisher exits normally.
    pygame.time.wait(500)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
