#!/usr/bin/env bash

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <BASE_DIR>"
    exit 1
fi

base_dir="$1"
while read dir; do
    echo "Processing ${dir}"
    ./make_gifs.py -i "${dir}" -o "${dir}.gif"
done < <(find "${base_dir}" ! -path "${base_dir}" -type d)
