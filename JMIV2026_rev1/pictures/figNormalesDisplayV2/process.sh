#!/bin/bash
# Process images in figNormalesDisplayV2:
#   1. Trim background and compress all PNG images
#   2. For each -II- and -PCA- set, create a combined image with a diagonal split:
#      - pixels above the diagonal (upper-left) → from the "4/45" variant
#      - pixels below the diagonal (lower-right) → from the "8/9"  variant

set -e
cd "$(dirname "$0")"

# ── Step 1: trim all images ──────────────────────────────────────────────────
echo "Trimming and compressing all PNG images..."
mogrify -trim -quality 85 *.png
echo "Done."

# ── Helper: diagonal composite via Python/Pillow ────────────────────────────
diagonal_combine() {
    local img_tl="$1"   # top-left  image (above the diagonal)
    local img_br="$2"   # bottom-right image (below the diagonal)
    local out="$3"      # output path

    python3 - "$img_tl" "$img_br" "$out" <<'PYEOF'
import sys
import numpy as np
from PIL import Image

img1 = Image.open(sys.argv[1]).convert('RGBA')
img2 = Image.open(sys.argv[2]).convert('RGBA')

# Resize img2 to match img1 if necessary
if img2.size != img1.size:
    img2 = img2.resize(img1.size, Image.LANCZOS)

W, H = img1.size

# Build pixel-coordinate grids
xs = np.arange(W, dtype=np.float64)
ys = np.arange(H, dtype=np.float64)
xx, yy = np.meshgrid(xs, ys)   # shape (H, W)

# Diagonal from top-left (0, 0) to bottom-right (W, H):
#   line equation: y = H*(x/W)
# "above" (upper-right region): y < H*(x/W)  → image 1
# "below" (lower-left region):  y >= H*(x/W) → image 2
above = yy < H * (xx / W)   # boolean mask, shape (H, W)

arr1 = np.array(img1)   # (H, W, 4)
arr2 = np.array(img2)   # (H, W, 4)

result = arr2.copy()
result[above] = arr1[above]

# Draw a thin black line along the diagonal (top-left to bottom-right)
from PIL import ImageDraw
out_img = Image.fromarray(result.astype(np.uint8))
draw = ImageDraw.Draw(out_img)
draw.line([(0, 0), (W - 1, H - 1)], fill=(0, 0, 0, 255), width=2)
out_img.save(sys.argv[3])
print(f"  → '{sys.argv[3]}' created ({W}x{H} px).")
PYEOF
}

# ── Step 2: create combined images ──────────────────────────────────────────

create_combined() {
    local tag="$1"          # e.g. "II" or "PCA"
    local suffix_tl="$2"    # suffix for the top-left  image (e.g. "45" or "4")
    local suffix_br="$3"    # suffix for the bottom-right image (e.g. "9"  or "8")

    for img_tl in *-${tag}-${suffix_tl}.png; do
        [ -f "$img_tl" ] || continue

        prefix="${img_tl%-${tag}-${suffix_tl}.png}"
        img_br="${prefix}-${tag}-${suffix_br}.png"

        if [ ! -f "$img_br" ]; then
            echo "WARNING: counterpart not found for '$img_tl' (expected '$img_br'). Skipping."
            continue
        fi

        out="${prefix}-${tag}-combined.png"
        echo "Creating '$out'  <-- '$img_tl' (upper-left) + '$img_br' (lower-right)..."
        diagonal_combine "$img_tl" "$img_br" "$out"
    done
}

echo ""
echo "Creating combined images for -II- sets..."
create_combined "II"  "45" "9"

echo ""
echo "Creating combined images for -PCA- sets..."
create_combined "PCA" "4"  "8"

echo ""
echo "All done."




