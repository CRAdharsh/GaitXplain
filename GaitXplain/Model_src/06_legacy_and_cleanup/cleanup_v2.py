from pathlib import Path
import shutil

OUTPUT_DIR = Path(
    "/Final Year Project/"
    "GAVD-main/stage2_test/v2_2sec"
)


def main():
    print("=" * 80)
    print("CLEAR GAVD V2 PREPROCESSED OUTPUT")
    print("=" * 80)
    print(f"Target: {OUTPUT_DIR}")

    if not OUTPUT_DIR.exists():
        print("Output directory does not exist. Nothing to remove.")
        return

    confirm = input("Delete this entire v2_2sec directory? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Cancelled. No files were modified.")
        return

    shutil.rmtree(OUTPUT_DIR)
    print("Deleted successfully.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "sequences").mkdir(parents=True, exist_ok=True)
    print("Recreated empty output directory and sequences folder.")


if __name__ == "__main__":
    main()

