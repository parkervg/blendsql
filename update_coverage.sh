#!/usr/bin/env bash
set -euo pipefail

# Load .env file if it exists
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# Validate required env vars
: "${GIST_TOKEN:?Set GIST_TOKEN in .env or environment}"
: "${GIST_ID:?Set GIST_ID in .env or environment}"

# 1. Run your GPU tests locally with tox
python -m tox

# 2. Combine coverage and generate JSON
python -m tox -e coverage

# 3. Extract the total percentage
TOTAL=$(python -c "import json; print(json.load(open('coverage.json'))['totals']['percent_covered_display'])")
echo "Total coverage: ${TOTAL}%"

# 4. Determine badge color (maps 50-90 range to red-green)
CONTENT=$(python3 -c "
import json

total = float('${TOTAL}')
min_range, max_range = 50, 90
ratio = max(0, min(1, (total - min_range) / (max_range - min_range)))

if ratio < 0.5:
    r, g = 255, int(255 * (ratio * 2))
else:
    r, g = int(255 * (1 - (ratio - 0.5) * 2)), 255
hex_color = f'{r:02x}{g:02x}00'

badge = {
    'schemaVersion': 1,
    'label': 'Coverage',
    'message': f'{total}%',
    'color': hex_color
}

print(json.dumps(badge))
")

# 5. Upload to the gist
curl -s -X PATCH \
  -H "Authorization: token ${GIST_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/gists/e24f1214fdff3ab086b829b5f01f85a8" \
  -d "$(jq -n --arg content "$CONTENT" '{files: {"covbadge.json": {content: $content}}}')"

echo "Badge updated: ${TOTAL}%"