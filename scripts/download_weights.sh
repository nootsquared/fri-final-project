#!/usr/bin/env bash
# download_weights.sh — Fetch HRNet-W32 and MEBOW orientation weights.
#
# Outputs two files in models/:
#   models/pose_hrnet_w32_256x192.pth   HRNet-W32 backbone (mmpose/COCO pose)
#   models/mebow_orientation.pth         MEBOW fine-tuned model
#
# SHA-256 checksums are printed after each download for reproducibility.
#
# Requirements: curl or wget, sha256sum (Linux) / shasum -a 256 (macOS)

set -euo pipefail

MODELS_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
mkdir -p "$MODELS_DIR"

# ---------------------------------------------------------------------------
# Checksum helper
# ---------------------------------------------------------------------------
checksum() {
    local file="$1"
    if command -v sha256sum &>/dev/null; then
        sha256sum "$file"
    else
        shasum -a 256 "$file"
    fi
}

# ---------------------------------------------------------------------------
# Download helper (curl with fallback to wget)
# ---------------------------------------------------------------------------
download() {
    local url="$1"
    local dest="$2"
    if [ -f "$dest" ] && [ "$(wc -c < "$dest")" -gt 1024 ]; then
        echo "[skip] $(basename "$dest") already exists — delete to re-download."
        return
    fi
    echo "[download] $(basename "$dest")"
    if command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$dest" "$url"
    elif command -v wget &>/dev/null; then
        wget -q --show-progress -O "$dest" "$url"
    else
        echo "ERROR: neither curl nor wget found." >&2
        exit 1
    fi
    # Abort if what we got is an HTML/text error page (< 1 MB is suspicious for a model)
    local size
    size="$(wc -c < "$dest")"
    if [ "$size" -lt 1048576 ]; then
        echo "ERROR: Downloaded file is only ${size} bytes — likely a redirect/error page." >&2
        echo "       Check the URL in this script and try again." >&2
        rm -f "$dest"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# 1. HRNet-W32 backbone (256×192, COCO pose, mmpose release)
#
#    Published by OpenMMLab at:
#      https://github.com/open-mmlab/mmpose
# ---------------------------------------------------------------------------
HRNET_URL="https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth"
HRNET_DEST="$MODELS_DIR/pose_hrnet_w32_256x192.pth"

echo ""
echo "=== HRNet-W32 backbone ==="
download "$HRNET_URL" "$HRNET_DEST"
echo "[sha256] $(checksum "$HRNET_DEST")"

# ---------------------------------------------------------------------------
# 2. MEBOW orientation model
#
#    Wu et al., "MEBOW: Monocular Estimation of Body Orientation In the Wild",
#    CVPR 2020.  https://github.com/ChenyanWu/MEBOW
#
#    The authors distribute weights via a request form / Google Drive link
#    listed in the MEBOW README.  Copy the direct download link here once
#    you have access, or place the file manually at:
#
#        models/mebow_orientation.pth
#
#    Uncomment and set MEBOW_URL once you have the correct link:
# ---------------------------------------------------------------------------
MEBOW_DEST="$MODELS_DIR/mebow_orientation.pth"

if [ -f "$MEBOW_DEST" ] && [ "$(wc -c < "$MEBOW_DEST")" -gt 1048576 ]; then
    echo ""
    echo "=== MEBOW orientation model ==="
    echo "[skip] mebow_orientation.pth already exists."
    echo "[sha256] $(checksum "$MEBOW_DEST")"
else
    echo ""
    echo "=== MEBOW orientation model ==="
    echo "  ACTION REQUIRED: MEBOW weights must be obtained from the authors."
    echo ""
    echo "  1. Visit https://github.com/ChenyanWu/MEBOW"
    echo "  2. Follow the README instructions to download the pretrained model."
    echo "  3. Save the file as:  models/mebow_orientation.pth"
    echo ""
    echo "  Without this file the orientation head will be randomly initialized"
    echo "  and yaw predictions will be meaningless."
fi

echo ""
echo "Done. Weight files:"
ls -lh "$MODELS_DIR"/*.pth 2>/dev/null || echo "(none found)"
