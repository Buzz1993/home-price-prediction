import json
from pathlib import Path

# Dynamically gets the directory of this script
current_dir = Path(__file__).resolve().parent

# Go up until you find the project root that contains the "data" folder
# Assuming this script lives in project_root/src/mcp/
ROOT_DIR = current_dir.parent.parent  # src/mcp -> src -> project_root

# Construct the file path to match exactly where your generator script saves it
METADATA_PATH = ROOT_DIR / "data" / "cleaned" / "search_metadata.json"

try:
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        SEARCH_METADATA = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Could not load search metadata. File missing at: {METADATA_PATH}. "
        f"Please run 'generate_search_metadata.py' first."
    )