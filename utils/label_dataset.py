import os
import argparse


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


DEFAULT_MAPPING = {
    'healthy': 'healthy',
    'pmk_laktasi': 'fmd_laktasi',
    'pmk_podal': 'fmd_podal',
    'pmk_oral': 'fmd_oral',
    'pmk_akut_general': 'fmd_akut',
}


def list_images(dirpath):
    return [f for f in sorted(os.listdir(dirpath)) if f.lower().endswith(IMAGE_EXTS)]


def rename_folder_images(folder, new_prefix, dry_run=True, start_index=1):
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        print(f"Skipping missing folder: {folder}")
        return 0

    files = list_images(folder)
    if not files:
        print(f"No images found in {folder}")
        return 0

    count = 0
    for i, fname in enumerate(files, start_index):
        src = os.path.join(folder, fname)
        _, ext = os.path.splitext(fname)
        new_name = f"{new_prefix}_{i}{ext.lower()}"
        dst = os.path.join(folder, new_name)

        if src == dst:
            count += 1
            continue

        if dry_run:
            print(f"[DRY] {src} -> {dst}")
        else:
            try:
                # If destination exists, append a suffix to avoid overwrite
                if os.path.exists(dst):
                    base, e = os.path.splitext(new_name)
                    j = 1
                    while os.path.exists(os.path.join(folder, f"{base}_{j}{e}")):
                        j += 1
                    dst = os.path.join(folder, f"{base}_{j}{e}")
                os.rename(src, dst)
                print(f"Renamed: {src} -> {dst}")
                count += 1
            except Exception as ex:
                print(f"Failed to rename {src}: {ex}")

    return count


def run_labeling(data_dir='dataset', mapping=None, dry_run=True):
    mapping = mapping or DEFAULT_MAPPING
    total = 0
    for folder_name, prefix in mapping.items():
        folder = os.path.join(data_dir, folder_name)
        n = rename_folder_images(folder, prefix, dry_run=dry_run)
        print(f"Processed {n} files in {folder_name}")
        total += n

    print(f"Total processed: {total}")
    return total


def main():
    parser = argparse.ArgumentParser(description='Label/rename dataset images by folder prefix')
    parser.add_argument('--data-dir', default='dataset', help='Dataset root folder')
    parser.add_argument('--apply', action='store_true', help='Actually perform rename (default is dry-run)')
    parser.add_argument('--start-index', type=int, default=1, help='Start index for numbering')
    parser.add_argument('--map', nargs='*', help='Custom mapping in form folder:prefix (overrides defaults)')

    args = parser.parse_args()

    mapping = DEFAULT_MAPPING.copy()
    if args.map:
        for item in args.map:
            if ':' in item:
                k, v = item.split(':', 1)
                mapping[k] = v

    print(f"Data dir: {args.data_dir}")
    print("Mapping:")
    for k, v in mapping.items():
        print(f"  {k} -> {v}")
    print("Dry run: not renaming files" if not args.apply else "Applying renames now")

    run_labeling(data_dir=args.data_dir, mapping=mapping, dry_run=not args.apply)


if __name__ == '__main__':
    main()
