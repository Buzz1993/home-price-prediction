#==============================
#search_metadata.py
#==============================
# search_metadata.py reads the dataset, extracts unique searchable values
# from only the columns listed in KEEP_COLUMNS, cleans them, removes duplicates,
# and saves them into search_metadata.json.
#
# Note: This metadata is created once for the entire dataset, not separately for each row.
'''
{
  "city": [
    "mumbai",
    "thane"
  ],
  "location": [
    "andheri",
    "goregaon"
  ],
  "amenities_mcp": [
    "cctv camera",
    "gymnasium",
    "swimming pool"
  ]
}
'''



import ast
import json
import re
from pathlib import Path
import pandas as pd

KEEP_COLUMNS = [
    "city",
    "location",
    "builder",
    "bhk_type",
    "property_type",
    "status",
    "furnish",
    "facing",
    "flooring",
    "ownership",
    "construction",
    "extra_rooms",
    "overlooking",
    "transportation_hubs_clean",
    "project_name",
    "nearest_mcp",
    "amenities_mcp",
    "features_mcp",
]

# Collection of explicit garbage values, loose symbols, and spreadsheet artifacts
JUNK_WORDS = {
    "#name?", "&", "&not", "and", "nan", "none", "null", "none of these", 
    ".", "..", "...", "....", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "00", ""
}


def is_valid_tag(text, column_name):
    """Filters out descriptions, paragraphs, dimensions, or garbage junk values."""
    text_lower = text.lower().strip()
    
    if text_lower in JUNK_WORDS:
        return False
        
    # Drop entries consisting entirely of lone single/double symbols or punctuation
    if len(text_lower) <= 2 and not text_lower.isalnum():
        return False
        
    # Drop pure loose numeric values (e.g., "000", "15")
    if text_lower.isdigit():
        return False

    # Apply specialized validation gates on text-heavy token columns
    if column_name in ["features_mcp", "nearest_mcp", "amenities_mcp", "project_name"]:
        
        # 1. Catch and drop paragraph-length user reviews and conversational text blocks.
        # Strings matching these contextual predicates are descriptions rather than filterable tags.
        narrative_keywords = ["offers", "boasts", "reviews", "praised", "criticize", "residents", "families", "located at", "situated in"]
        if any(kw in text_lower for kw in narrative_keywords):
            return False
            
        # 2. Filter out raw material dimensions and tile layouts (e.g., "800 mm x 800 mm vitrified tiles")
        if re.search(r'\d+\s*(mm|in|inch|ft|x|\*)\s*x?\s*\d*', text_lower):
            return False
            
        # 3. String Word and Length Threshold Gating
        # Keep clean UI filters bounded to crisp concise phrases (max 5 words or 50 characters)
        if len(text_lower.split()) > 5 or len(text_lower) > 50:
            return False
            
    return True


def clean_and_split(text, column_name):
    """Parses, cleans, and splits raw string inputs into clean keyword arrays."""
    if not isinstance(text, str):
        text = str(text)
        
    # Clean out stray HTML break fragments and explicit carriage line returns
    text = text.replace("<br>", " ").replace("\n", " ").replace("\r", " ")
    text = text.strip()
    
    if not text or text.lower() in ("nan", "none", "null", ""):
        return []

    # Safely evaluate and process stringified Python lists: "['Amenity 1', 'Amenity 2']"
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                extracted = []
                for item in parsed:
                    extracted.extend(clean_and_split(item, column_name))
                return extracted
        except (ValueError, SyntaxError):
            pass

    # Unify different column row-delimiters into uniform commas and convert to lower-case
    normalized = text.replace("|", ",").replace(";", ",").lower()

    extracted_tokens = []
    for val in normalized.split(","):
        cleaned_val = val.strip()
        
        # Eliminate dangling peripheral boundaries (e.g., quotes, bullet indicators, raw slashes, or artifacts)
        cleaned_val = re.sub(r'^[\"\’\‘\“\”\„\'\-\&\*\s\.\,\\\/âï¿½•_]+', '', cleaned_val)
        cleaned_val = re.sub(r'[\"\’\‘\“\”\„\'\-\&\*\s\.\,\\\/âï¿½•_]+$', '', cleaned_val)
        
        # Conflate duplicate internal white spaces into a single blank character space
        cleaned_val = " ".join(cleaned_val.split())
        
        # Standardize direction space delimiters uniformly (e.g., "north  -  east" -> "north-east")
        cleaned_val = re.sub(r'\s*-\s*', '-', cleaned_val)
        
        # Apply validation logic gate parameters
        if cleaned_val and is_valid_tag(cleaned_val, column_name):
            extracted_tokens.append(cleaned_val)
            
    return extracted_tokens


def extract_values(series, column_name):
    """Extracts, deduplicates, and sorts unique tokens from a single dataframe column."""
    values = set()
    for item in series.dropna():
        extracted_tags = clean_and_split(item, column_name)
        values.update(extracted_tags)
    return sorted(list(values))


def build_search_metadata(df):
    """Compiles unique, sorted values for all target metadata schema columns."""
    metadata = {}
    for col in KEEP_COLUMNS:
        if col not in df.columns:
            print(f"⚠️ Warning: Target column '{col}' missing from source dataframe schema. Skipping.")
            continue

        values = extract_values(df[col], col)
        metadata[col] = values
        print(f"✅ {col}: {len(values)} pristine sorted lookup entries successfully compiled")

    return metadata


def main():
    """Main pipeline execution to read data, generate metadata, and save to JSON."""
    # Dynamically navigate up 3 levels to locate your file architecture root block
    root_path = Path(__file__).resolve().parent.parent.parent

    input_file = root_path / "data" / "cleaned" / "final_combined_mcp_data.csv"
    output_file = root_path / "data" / "cleaned" / "search_metadata.json"

    print(f"Reading target file source from path: {input_file}")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(
            f"❌ Error: Failed to find target data at location path: {input_file}.\n"
            f"Please verify that your raw pipeline engine has executed successfully first!"
        )
        return

    # Trigger metadata parsing pipeline architecture
    metadata = build_search_metadata(df)

    # Export to clean pretty-printed output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n🚀 Clean metadata lookup schema successfully generated and saved to: {output_file}")


if __name__ == "__main__":
    main()