from pathlib import Path
import shutil

OUTPUT_DIR = Path(
    "/Final Year Project/"
    "GAVD-main/stage2_test/v2_2sec"
)

print("=" * 80)
print("V2 PREPROCESSED DATASET REMOVAL")
print("=" * 80)

print(f"\nTarget directory:")
print(OUTPUT_DIR)

if not OUTPUT_DIR.exists():
    print("\nDirectory does not exist.")
    print("Nothing to delete.")
    raise SystemExit(0)

print("\nWARNING:")
print("This will permanently delete the contents of:")
print(OUTPUT_DIR)
print("\nIt will NOT delete:")
print("- original downloaded videos")
print("- GAVD annotations")
print("- stage2_test/test_videos.csv")
print("- V1 Dataset")
print("- files outside v2_2sec")

answer = input("\nType DELETE_V2 to continue: ").strip()

if answer != "DELETE_V2":
    print("\nDeletion cancelled.")
    raise SystemExit(0)

print("\nDeleting...")

shutil.rmtree(OUTPUT_DIR)

print("\n" + "=" * 80)
print("OLD V2 DATASET DELETED")
print("=" * 80)
print(f"\nRemoved:\n{OUTPUT_DIR}")