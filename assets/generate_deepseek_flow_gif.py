"""Generate DeepSeek V4 inspired LLM inference flowchart pixel GIF."""

from __future__ import annotations

from PIL import Image, ImageDraw

W, H = 960, 140
PX = 3
FRAMES_PER_STEP = 5
HOLD_FRAMES = 8

# DeepSeek-inspired palette
BG = (10, 14, 22)
GRID = (22, 30, 44)
NODE_OFF = (26, 36, 52)
NODE_OFF_BORDER = (42, 56, 78)
NODE_DONE = (36, 72, 120)
NODE_ACTIVE = (77, 107, 254)
NODE_GLOW = (120, 160, 255)
ARROW_OFF = (38, 50, 68)
ARROW_ON = (90, 180, 255)
PULSE = (0, 212, 170)
LABEL = (180, 200, 230)

# Simplified DeepSeek-V4 inference pipeline (flowchart nodes)
STAGES = [
    ("IN", "Input Tokens"),
    ("EMB", "Token Embedding"),
    ("A1", "Layer 1 · MLA Attention"),
    ("M1", "Layer 1 · MoE FFN"),
    ("A2", "Layer 2 · MLA Attention"),
    ("M2", "Layer 2 · MoE FFN"),
    ("A3", "Layer 3 · MLA Attention"),
    ("M3", "Layer 3 · MoE FFN"),
    ("N", "RMSNorm"),
    ("H", "LM Head"),
    ("OUT", "Next Token"),
]


def snap(v: int) -> int:
    return v - (v % PX)


def draw_box(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    state: str,
    tag: str,
    subtitle: str,
) -> None:
    x, y, w, h = snap(x), snap(y), snap(w), snap(h)

    if state == "active":
        fill, border, glow = NODE_ACTIVE, NODE_GLOW, True
    elif state == "done":
        fill, border, glow = NODE_DONE, NODE_ACTIVE, False
    else:
        fill, border, glow = NODE_OFF, NODE_OFF_BORDER, False

    if glow:
        for dx, dy in [(-PX, 0), (PX, 0), (0, -PX), (0, PX)]:
            draw.rectangle([x + dx - PX, y + dy - PX, x + w + dx + PX, y + h + dy + PX], fill=PULSE)

    draw.rectangle([x, y, x + w, y + h], fill=fill, outline=border, width=PX)

    # pixel tag (centered abbreviation)
    tx = x + w // 2 - len(tag) * PX
    ty = y + h // 2 - PX * 2
    for i, ch in enumerate(tag):
        cx = tx + i * PX * 3
        draw.rectangle([cx, ty, cx + PX * 2, ty + PX * 2], fill=LABEL if state != "off" else (90, 100, 120))

    # tiny progress bar inside active node
    if state == "active":
        bar_y = y + h - PX * 3
        draw.rectangle([x + PX * 2, bar_y, x + w - PX * 2, bar_y + PX], fill=NODE_GLOW)


def draw_arrow(draw: ImageDraw.ImageDraw, x1: int, y: int, x2: int, lit: bool) -> None:
    color = ARROW_ON if lit else ARROW_OFF
    mid_y = snap(y)
    draw.rectangle([x1, mid_y, x2, mid_y + PX], fill=color)
    # arrow head
    tip = x2
    draw.polygon([(tip, mid_y - PX), (tip + PX * 2, mid_y + PX // 2), (tip, mid_y + PX * 2)], fill=color)


def draw_title(draw: ImageDraw.ImageDraw, active_idx: int) -> None:
    draw.rectangle([0, 0, W, PX * 5], fill=(14, 20, 32))
    # title bar blocks
    for i in range(18):
        c = NODE_ACTIVE if (i + active_idx) % 4 == 0 else (40, 55, 80)
        draw.rectangle([PX * 4 + i * PX * 2, PX, PX * 4 + i * PX * 2 + PX, PX * 4], fill=c)
    draw.rectangle([PX * 3, PX * 5, W - PX * 3, PX * 6], fill=GRID)


def draw_status(draw: ImageDraw.ImageDraw, active_idx: int) -> None:
    _, subtitle = STAGES[min(active_idx, len(STAGES) - 1)]
    bar_y = H - PX * 7
    draw.rectangle([PX * 3, bar_y, W - PX * 3, bar_y + PX * 5], fill=(16, 22, 34), outline=NODE_OFF_BORDER, width=PX)
    # pixel activity indicator + stage name blocks
    draw.rectangle([PX * 5, bar_y + PX * 2, PX * 8, bar_y + PX * 4], fill=PULSE if active_idx < len(STAGES) else NODE_GLOW)
    sx = PX * 11
    for i, _ in enumerate(subtitle):
        lit = i <= (active_idx * 2) % len(subtitle)
        c = LABEL if lit else (55, 65, 85)
        draw.rectangle([sx + i * PX, bar_y + PX * 2, sx + i * PX + PX - 1, bar_y + PX * 4], fill=c)


def layout_nodes() -> list[tuple[int, int, int, int]]:
    n = len(STAGES)
    margin = PX * 4
    gap = PX * 2
    usable = W - margin * 2 - gap * (n - 1)
    box_w = usable // n
    box_h = PX * 14
    y = PX * 10
    boxes = []
    x = margin
    for _ in STAGES:
        boxes.append((x, y, box_w, box_h))
        x += box_w + gap
    return boxes


def make_frame(active_idx: int, pulse: int) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw_title(draw, active_idx)

    boxes = layout_nodes()
    for i, ((x, y, bw, bh), (tag, sub)) in enumerate(zip(boxes, STAGES)):
        if i < active_idx:
            state = "done"
        elif i == active_idx:
            state = "active"
        else:
            state = "off"
        draw_box(draw, x, y, bw, bh, state, tag, sub)

        if i < len(boxes) - 1:
            nx, ny, nw, nh = boxes[i + 1]
            ax1 = x + bw + PX
            ax2 = nx - PX
            ay = y + bh // 2
            draw_arrow(draw, ax1, ay, ax2, lit=i < active_idx)

    # forward pass pulse dot traveling on active arrow
    if 0 <= active_idx < len(boxes) - 1:
        x, y, bw, bh = boxes[active_idx]
        nx, _, _, _ = boxes[active_idx + 1]
        t = pulse / FRAMES_PER_STEP
        px = int(x + bw + (nx - x - bw) * t)
        py = snap(y + bh // 2 - PX)
        draw.rectangle([px, py, px + PX * 2, py + PX * 2], fill=PULSE)

    draw_status(draw, active_idx)
    return img


def build_frames() -> list[Image.Image]:
    frames: list[Image.Image] = []
    for step in range(len(STAGES)):
        for pulse in range(FRAMES_PER_STEP):
            frames.append(make_frame(step, pulse))
    for _ in range(HOLD_FRAMES):
        frames.append(make_frame(len(STAGES) - 1, FRAMES_PER_STEP - 1))
    return frames


def main() -> None:
    frames = build_frames()
    out = "assets/deepseek-v4-flow.gif"
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=True,
    )
    print(f"Saved {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
