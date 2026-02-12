#!/usr/bin/env bash
set -euo pipefail

# Load .env file if it exists
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

: "${GIST_TOKEN:?Set GIST_TOKEN in .env or environment}"
: "${GIST_ID:?Set GIST_ID in .env or environment}"

UPLOAD_ONLY=false
for arg in "$@"; do
  case $arg in
    --upload-only) UPLOAD_ONLY=true ;;
  esac
done

if [ "$UPLOAD_ONLY" = false ]; then
  python -m tox -e coverage
fi

TOTAL=$(python -c "import json; print(json.load(open('coverage.json'))['totals']['percent_covered_display'])")
echo "Total coverage: ${TOTAL}%"

CONTENT=$(python3 -c "
import json, colorsys

total = float('${TOTAL}')
min_range, max_range = 50, 90
ratio = max(0, min(1, (total - min_range) / (max_range - min_range)))

hue = ratio * 120 / 360
r, g, b = colorsys.hls_to_rgb(hue, 0.65, 0.5)
hex_color = f'{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

badge = {
    'schemaVersion': 1,
    'label': 'Coverage',
    'message': f'{round(total)}%',
    'color': hex_color
}
print(json.dumps(badge))
")

curl -s -X PATCH \
  -H "Authorization: token ${GIST_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/gists/${GIST_ID}" \
  -d "$(jq -n --arg content "$CONTENT" '{files: {"covbadge.json": {content: $content}}}')"

echo "Badge updated: ${TOTAL}%"