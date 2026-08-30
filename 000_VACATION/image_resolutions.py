from collections import Counter
from pathlib import Path
from PIL import Image


def analyze_image_folder(folder_path):
  # Define common image extensions to check
  supported_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}
  root_dir = Path(folder_path)

  resolution_counts = Counter()
  total_images = 0
  failed_images = 0

  print(f"Scanning folder (including subfolders): {root_dir.resolve()}\n")
  print(f"{'Image Path':<50} | {'Resolution (WxH)':<18}")
  print("-" * 71)

  # rglob('*') recursively finds all files in subfolders
  for file_path in root_dir.rglob("*"):
    if file_path.suffix.lower() in supported_extensions:
      try:
        with Image.open(file_path) as img:
          width, height = img.size
          res_str = f"{width}x{height}"
          resolution_counts[res_str] += 1
          total_images += 1

          # Display relative path for a cleaner output
          rel_path = file_path.relative_to(root_dir)
          print(f"{str(rel_path):<50} | {res_str:<18}")
      except Exception as e:
        failed_images += 1
        print(f"Could not read {file_path.name}: {e}")

  print("\n" + "=" * 71)
  print("SUMMARY REPORT")
  print("=" * 71)
  print(f"Total successfully scanned images: {total_images}")
  if failed_images > 0:
    print(f"Skipped/Corrupted files: {failed_images}")

  print("\nResolution Breakdown:")
  for res, count in resolution_counts.most_common():
    percentage = (count / total_images * 100) if total_images > 0 else 0
    print(f"  - {res}: {count} image(s) ({percentage:.1f}%)")


if __name__ == "__main__":
  # Set your target folder here. Use "." for the folder where the script is located,
  # or provide an absolute path like r"C:\Users\Name\Pictures"
  target_folder = "C:\\repo\\VACATION\\images\\"
  analyze_image_folder(target_folder)