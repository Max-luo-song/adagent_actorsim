from pathlib import Path
from PIL import Image, ImageDraw


def compose_five(root):
    root = Path(root)
    canvas = Image.new("RGB", (1440, 712), "white")
    draw = ImageDraw.Draw(canvas)

    top = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
    bottom = ["CAM_SIDE_LEFT", "CAM_SIDE_RIGHT"]

    for i, name in enumerate(top):
        with Image.open(root / (name + ".png")) as image:
            frame = image.convert("RGB").resize(
                (480, 320), Image.Resampling.LANCZOS
            )
        x = i * 480
        draw.text((x + 8, 8), name, fill="black")
        canvas.paste(frame, (x, 28))

    for i, name in enumerate(bottom):
        with Image.open(root / (name + ".png")) as image:
            frame = image.convert("RGB").resize(
                (720, 324), Image.Resampling.LANCZOS
            )
        x = i * 720
        draw.text((x + 8, 368), name, fill="black")
        canvas.paste(frame, (x, 388))

    path = root / "five_views_two_rows.png"
    canvas.save(path)
    canvas.save(root / "five_views.png")
    return path
