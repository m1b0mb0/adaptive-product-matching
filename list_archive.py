"""Show file names in the official WDC Products 80pair.zip archive."""
import argparse
from zipfile import ZipFile

p = argparse.ArgumentParser()
p.add_argument("zip_path")
args = p.parse_args()
with ZipFile(args.zip_path) as z:
    for info in z.infolist():
        print(f"{info.file_size:>11}  {info.filename}")
