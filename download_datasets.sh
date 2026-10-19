#!/usr/bin/env bash
# Explicit server-side acquisition. Requires curl, unzip, python3; no Kaggle CLI.
set -euo pipefail
DATA_ROOT=${CLEAR_DATA_ROOT:-data/raw}
ARCHIVE_DIR=${CLEAR_ARCHIVE_DIR:-data/archives}
DATASET=all
while (($#)); do
    case "$1" in
        --dataset) DATASET=$2; shift 2 ;;
        --data-root) DATA_ROOT=$2; shift 2 ;;
        --archive-dir) ARCHIVE_DIR=$2; shift 2 ;;
        -h|--help) printf 'Usage: bash download_datasets.sh [--dataset indiana|padchest|chexpert|all] [--data-root PATH] [--archive-dir PATH]\nAuth: optional KAGGLE_USERNAME/KAGGLE_KEY, KAGGLE_API_TOKEN, or ~/.kaggle/kaggle.json.\n'; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
for cmd in curl unzip python3; do command -v "$cmd" >/dev/null || { printf 'Missing %s; install it manually.\n' "$cmd" >&2; exit 1; }; done
case "$DATASET" in all) DATASETS=(indiana padchest chexpert) ;; indiana|padchest|chexpert) DATASETS=("$DATASET") ;; *) printf 'Unknown dataset\n' >&2; exit 2 ;; esac
mkdir -p "$ARCHIVE_DIR" "$DATA_ROOT"
# Keep secrets out of process arguments and logs; curl reads a private config.
AUTH_CONFIG=$(mktemp)
chmod 600 "$AUTH_CONFIG"
trap 'rm -f -- "$AUTH_CONFIG"' EXIT
python3 - "$AUTH_CONFIG" <<'PY'
import json, os, pathlib, sys
username, key = os.getenv('KAGGLE_USERNAME'), os.getenv('KAGGLE_KEY')
token = os.getenv('KAGGLE_API_TOKEN')
credentials = pathlib.Path(os.getenv('KAGGLE_CONFIG_DIR', str(pathlib.Path.home()/'.kaggle'))) / 'kaggle.json'
if not token and not (username and key) and credentials.is_file():
    data = json.loads(credentials.read_text())
    username, key = data.get('username'), data.get('key')
with open(sys.argv[1], 'w') as output:
    if token:
        output.write('header = ' + json.dumps('Authorization: Bearer ' + token) + '\n')
    elif username and key:
        output.write('user = ' + json.dumps(username + ':' + key) + '\n')
PY
for name in "${DATASETS[@]}"; do
    case "$name" in
        indiana) SLUG=raddar/chest-xrays-indiana-university; FILE=chest-xrays-indiana-university.zip ;;
        padchest) SLUG=raddar/padchest-chest-xrays-sample; FILE=padchest-chest-xrays-sample.zip ;;
        chexpert) SLUG=ashery/chexpert; FILE=chexpert.zip ;;
    esac
    ZIP="$ARCHIVE_DIR/$FILE"
    PART="$ZIP.part"
    DEST="$DATA_ROOT/$name"
    if [[ -f "$DEST/.clear-extraction-complete.json" ]]; then
        printf '%s already extracted; keeping the existing directory. Remove its marker explicitly to revalidate/extract.\n' "$name"
        continue
    fi
    if [[ -f "$ZIP" ]]; then
        if ! unzip -tq "$ZIP" >/dev/null; then
            printf 'Archive failed integrity check: %s. Preserve/rename it and retry; it will not be overwritten.\n' "$ZIP" >&2; exit 1
        fi
    else
        printf 'Downloading %s (supports resume if Kaggle/CDN supports byte ranges)\n' "$SLUG"
        set +e
        STATUS=$(curl --config "$AUTH_CONFIG" -fL -C - --retry 4 --retry-delay 5 --connect-timeout 30 \
            -w '%{http_code}' -o "$PART" "https://www.kaggle.com/api/v1/datasets/download/$SLUG")
        CODE=$?
        set -e
        if ((CODE != 0)); then
            printf 'curl failed (%s), HTTP %s. Partial archive preserved at %s.\n' "$CODE" "$STATUS" "$PART" >&2
            case "$STATUS" in
                401|403) printf 'Kaggle authentication/access denied. Accept dataset terms in your account; configure Kaggle credentials, then retry.\n' >&2 ;;
                404) printf 'Dataset endpoint unavailable or hidden by access restrictions; verify the Kaggle dataset page.\n' >&2 ;;
                416) printf 'Range rejected: partial may already be complete. Validate with unzip -t; otherwise rename .part and restart explicitly.\n' >&2 ;;
                *) printf 'If curl reports error 33, the server does not support resume. Rename the .part file and restart explicitly. Check available disk space and network access.\n' >&2 ;;
            esac
            exit 1
        fi
        if ! unzip -tq "$PART" >/dev/null; then
            printf 'Response is not a valid ZIP (possibly a Kaggle login/terms page or truncated transfer). Check access in browser; partial preserved.\n' >&2; exit 1
        fi
        mv -- "$PART" "$ZIP"
    fi
    # Validate member paths before unzip. In-place extraction avoids a second image tree.
    python3 - "$ZIP" <<'PY'
import pathlib, stat, sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    for info in z.infolist():
        path = pathlib.PurePosixPath(info.filename.replace('\\', '/'))
        if path.is_absolute() or '..' in path.parts or stat.S_ISLNK(info.external_attr >> 16):
            raise SystemExit('Unsafe archive member: ' + info.filename)
PY
    mkdir -p "$DEST"
    # -n resumes extraction without replacing already written files. Verify sizes/CRC below;
    # a file interrupted during extraction is rejected with a precise repair instruction.
    unzip -nq "$ZIP" -d "$DEST"
    python3 - "$ZIP" "$DEST" <<'PY'
import json, pathlib, sys, zipfile, zlib
archive, dest = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
with zipfile.ZipFile(archive) as z:
    for info in z.infolist():
        if info.is_dir():
            continue
        target = dest / info.filename
        crc, size = 0, 0
        with target.open('rb') as handle:
            for block in iter(lambda: handle.read(8*1024*1024), b''):
                crc, size = zlib.crc32(block, crc), size + len(block)
        if size != info.file_size or crc != info.CRC:
            raise SystemExit(f'Extracted file corrupt/incomplete: {target}. Remove ONLY this file explicitly, then rerun to extract it again.')
    count = len(z.infolist())
(dest/'.clear-extraction-complete.json').write_text(json.dumps({'archive': str(archive.resolve()), 'archive_bytes': archive.stat().st_size, 'members': count, 'crc_verified': True}, indent=2))
PY
    printf 'Ready: %s. Archive retained for integrity/resume; no second extracted copy created.\n' "$DEST"
done
