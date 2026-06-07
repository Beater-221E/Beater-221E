"""Generate Astral Express inspired pixel train GIF for GitHub profile."""

from PIL import Image, ImageDraw

W, H = 960, 120
FRAMES = 24
SCALE = 4  # pixel size


def draw_starfield(draw: ImageDraw.ImageDraw, frame: int) -> None:
    import random
    rng = random.Random(42)
    for _ in range(60):
        x = rng.randint(0, W // SCALE - 1) * SCALE
        y = rng.randint(0, (H // 2) // SCALE) * SCALE
        twinkle = (frame + rng.randint(0, 8)) % 12
        color = (255, 255, 255) if twinkle > 2 else (120, 120, 180)
        draw.rectangle([x, y, x + SCALE - 1, y + SCALE - 1], fill=color)


def draw_tracks(draw: ImageDraw.ImageDraw) -> None:
    y = H - SCALE * 3
    draw.rectangle([0, y, W, y + SCALE - 1], fill=(80, 70, 100))
    for x in range(0, W, SCALE * 3):
        draw.rectangle([x, y + SCALE, x + SCALE, y + SCALE * 2 - 1], fill=(140, 130, 160))


def draw_train(draw: ImageDraw.ImageDraw, offset_x: int) -> None:
    """Pixel-art space express train (original, HSR-inspired palette)."""
    px = SCALE
    base_y = H - px * 10
    x = offset_x

    # locomotive body
    body = (230, 225, 240)
    gold = (220, 180, 90)
    window = (120, 210, 255)
    accent = (160, 120, 220)
    dark = (60, 50, 90)

    parts = [
        # (x_off, y_off, w, h, color)
        (0, 2, 10, 4, body),
        (10, 3, 6, 3, body),
        (0, 1, 10, 1, gold),
        (10, 2, 6, 1, gold),
        (1, 3, 2, 2, window),
        (4, 3, 2, 2, window),
        (7, 3, 2, 2, window),
        (11, 4, 2, 1, window),
        (13, 4, 2, 1, window),
        (2, 6, 2, 2, accent),
        (6, 6, 2, 2, accent),
        (0, 6, 16, 1, dark),
        (15, 4, 2, 3, gold),  # nose
    ]
    for xo, yo, w, h, color in parts:
        draw.rectangle(
            [x + xo * px, base_y + yo * px, x + (xo + w) * px - 1, base_y + (yo + h) * px - 1],
            fill=color,
        )

    # passenger cars
    for car in range(2):
        cx = x + (18 + car * 9) * px
        draw.rectangle([cx, base_y + 2 * px, cx + 8 * px - 1, base_y + 6 * px - 1], fill=body)
        draw.rectangle([cx, base_y + 2 * px, cx + 8 * px - 1, base_y + 3 * px - 1], fill=gold)
        for wx in (1, 4):
            draw.rectangle(
                [cx + wx * px, base_y + 3 * px, cx + (wx + 2) * px - 1, base_y + 5 * px - 1],
                fill=window,
            )
        draw.rectangle([cx, base_y + 6 * px, cx + 8 * px - 1, base_y + 7 * px - 1], fill=dark)


def make_frame(frame: int) -> Image.Image:
    img = Image.new("RGB", (W, H), (18, 12, 38))
    draw = ImageDraw.Draw(img)
    draw_starfield(draw, frame)
    draw_tracks(draw)
    # train scrolls right; wrap around
    travel = (frame * (W // FRAMES + 20)) % (W + 200)
    draw_train(draw, W - travel)
    return img


def main() -> None:
    frames = [make_frame(i) for i in range(FRAMES)]
    out = "assets/astral-express-train.gif"
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
        optimize=True,
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
