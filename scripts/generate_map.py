"""Render the static office-location map used on the contact page.

The contact page used to end in an `ipyleaflet` cell. It pulled 4.5 MB of
JavaScript from three CDNs -- `require.js`, `@jupyter-widgets/html-manager`
pinned at `@*` and `jupyter-leaflet` at `^0.17`, both floating -- and then
threw `TypeError: Cannot read properties of undefined (reading
'invalidateSize')` inside the widget's layout pass, so the live page showed
empty space where the map should be. Two unpinned CDN ranges meant any
upstream release could break it again, and the widget had no static fallback.

The office is not going to move, so the map does not need a runtime at all.
This script stitches OpenStreetMap tiles into a single PNG once and commits
it; the page then carries one image and a link through to OSM for anyone who
wants to pan and zoom. Run it again only if the marker or framing changes::

    poetry run python scripts/generate_map.py

Tiles are cached under `data/osm_tiles/` so re-runs stay off OSM's servers.
The OpenStreetMap tile usage policy requires an identifying User-Agent and
visible attribution: the credit is burned into the image's bottom-right
corner, since the PNG can be opened on its own, and repeated as a caption
link on the page.
"""

import math
import sys
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont

try:
    from to_quarto.utils import ROOT_DIR
except ImportError:  # invoked as `python scripts/generate_map.py` from the root
    ROOT_DIR = Path(__file__).parent.parent.absolute()

OUTPUT_FILE = Path(ROOT_DIR).joinpath("static", "office_map.png")
TILE_CACHE = Path(ROOT_DIR).joinpath("data", "osm_tiles")

# Institut fuer Theoretische Physik, Philosophenweg 19. Verified against
# Nominatim rather than eyeballed: the building way is at 49.41505, 8.69844.
CENTER_LAT = 49.41505
CENTER_LON = 8.69844

# The image is drawn at twice its display size so it stays sharp on retina
# screens. 900 px is the site's `body-width`, so the map spans the text
# column exactly; zoom 17 at 2x puts roughly 1.4 km across that width, which
# is enough to place Philosophenweg against the Neckar and the Altstadt.
DISPLAY_WIDTH = 900
DISPLAY_HEIGHT = 400
SCALE = 2
ZOOM = 17

WIDTH = DISPLAY_WIDTH * SCALE
HEIGHT = DISPLAY_HEIGHT * SCALE

TILE_SIZE = 256
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
USER_AGENT = (
    "tristanbereau.com static map builder "
    "(scripts/generate_map.py; bereau@thphys.uni-heidelberg.de)"
)

ATTRIBUTION = "© OpenStreetMap contributors"
MARKER_COLOR = (57, 114, 158)  # $link-color in theme.scss
MARKER_RADIUS = 11 * SCALE

# A street map is flat colour over a handful of fills, so a palette holds it
# without visible loss: truecolour PNG is 797 kB, 128 colours is 254 kB and
# indistinguishable at 1x or 2x. Dithering is off because it speckles the
# large uniform greens and greys and costs back most of the saving.
PALETTE_COLORS = 128

# Whichever of these the machine has. PIL's built-in bitmap font does not
# scale, so it renders illegibly small on a 2x canvas; the fallback is only
# there so the script still produces an image rather than failing outright.
FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def lonlat_to_pixels(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    """Web Mercator, in pixels from the top-left of the world at `zoom`."""
    n = TILE_SIZE * 2**zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def fetch_tile(x: int, y: int, zoom: int) -> Image.Image:
    cached = TILE_CACHE.joinpath(f"{zoom}_{x}_{y}.png")
    if cached.exists():
        return Image.open(cached).convert("RGB")

    url = TILE_URL.format(z=zoom, x=x, y=y)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        payload = response.read()

    TILE_CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(payload)
    return Image.open(BytesIO(payload)).convert("RGB")


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_marker(canvas: Image.Image) -> None:
    """A filled dot with a white ring, centred on the building.

    A teardrop pin would point at its own tip and sit above the coordinate;
    a concentric dot reads unambiguously at both 1x and 2x, and needs no
    external icon asset.
    """
    draw = ImageDraw.Draw(canvas)
    cx, cy = WIDTH // 2, HEIGHT // 2
    ring = MARKER_RADIUS + 3 * SCALE
    draw.ellipse(
        (cx - ring, cy - ring, cx + ring, cy + ring),
        fill=(255, 255, 255),
        outline=(0, 0, 0, 40),
    )
    draw.ellipse(
        (cx - MARKER_RADIUS, cy - MARKER_RADIUS, cx + MARKER_RADIUS, cy + MARKER_RADIUS),
        fill=MARKER_COLOR,
    )


def draw_attribution(canvas: Image.Image) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = load_font(11 * SCALE)
    padding = 4 * SCALE

    left, top, right, bottom = draw.textbbox((0, 0), ATTRIBUTION, font=font)
    text_w, text_h = right - left, bottom - top
    box_w, box_h = text_w + 2 * padding, text_h + 2 * padding
    box_x, box_y = WIDTH - box_w, HEIGHT - box_h

    # OSM's own tiles put the credit on a translucent white plate for the
    # same reason: the underlying map is light but not uniformly so.
    draw.rectangle((box_x, box_y, WIDTH, HEIGHT), fill=(255, 255, 255, 190))
    draw.text(
        (box_x + padding - left, box_y + padding - top),
        ATTRIBUTION,
        font=font,
        fill=(60, 60, 60),
    )


def build_map() -> Image.Image:
    center_x, center_y = lonlat_to_pixels(CENTER_LON, CENTER_LAT, ZOOM)
    left = center_x - WIDTH / 2
    top = center_y - HEIGHT / 2

    first_tile_x = math.floor(left / TILE_SIZE)
    first_tile_y = math.floor(top / TILE_SIZE)
    last_tile_x = math.floor((left + WIDTH) / TILE_SIZE)
    last_tile_y = math.floor((top + HEIGHT) / TILE_SIZE)

    canvas = Image.new("RGB", (WIDTH, HEIGHT), (233, 229, 220))
    for tile_x in range(first_tile_x, last_tile_x + 1):
        for tile_y in range(first_tile_y, last_tile_y + 1):
            try:
                tile = fetch_tile(tile_x, tile_y, ZOOM)
            except (HTTPError, URLError, OSError) as error:
                # One missing tile leaves a blank square rather than losing
                # the whole map, but it should not pass silently.
                print(
                    f"warning: tile {ZOOM}/{tile_x}/{tile_y} failed ({error})",
                    file=sys.stderr,
                )
                continue
            canvas.paste(
                tile,
                (
                    int(tile_x * TILE_SIZE - left),
                    int(tile_y * TILE_SIZE - top),
                ),
            )

    draw_marker(canvas)
    draw_attribution(canvas)
    return canvas


def main() -> None:
    canvas = build_map()
    palette = canvas.quantize(
        colors=PALETTE_COLORS, method=Image.MEDIANCUT, dither=Image.NONE
    )
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    palette.save(OUTPUT_FILE, "PNG", optimize=True)
    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"wrote {OUTPUT_FILE.relative_to(ROOT_DIR)} ({size_kb:.0f} kB)")


if __name__ == "__main__":
    main()
