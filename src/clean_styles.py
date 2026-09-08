import csv
from pathlib import Path

RAW_FILE =Path("data/raw/fashion-product-images-small/styles.csv")
CLEAN_FILE = Path("data/processed/styles_clean.csv")

EXPECTED_COLUMNS = [
    "id",
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "season",
    "year",
    "usage",
    "productDisplayName"
]

def clean_styles_csv():
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    malformed_rows = 0
    short_rows = 0

    with open(RAW_FILE, "r", encoding="utf-8", newline="") as infile:
        reader = csv.reader(infile)
        header = next(reader)

        if header != EXPECTED_COLUMNS:
            raise ValueError(
                f"Unexpected header.\n"
                f"Expected: {EXPECTED_COLUMNS}\n"
                f"Found: {header}"
            )
        with open(CLEAN_FILE, "w", encoding="utf-8", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(EXPECTED_COLUMNS)
            for row in reader:
                if len(row) > len(EXPECTED_COLUMNS):
                    malformed_rows += 1
                    row = row[:9] + [",".join(row[9:])]

                elif len(row) < len(EXPECTED_COLUMNS):
                    short_rows += 1
                    print(f"Warning: row has only {len(row)} columns: {row}")
                writer.writerow(row)
    print("Cleaning completed.")
    print(f"Malformed rows repaired: {malformed_rows}")
    print(f"Short rows found: {short_rows}")
    print(f"Clean file Created: {CLEAN_FILE}")

if __name__ == "__main__":
    clean_styles_csv()