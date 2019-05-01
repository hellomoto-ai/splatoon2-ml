#!/usr/bin/env bash

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <BASE_DIR>"
    exit 1
fi

base_dir="$1"

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
script="${this_dir}/make_gifs.py"
while read dir; do
    echo "Processing ${dir}"
    "${script}" -i "${dir}" -o "${dir}.gif"
done < <(find "${base_dir}" ! -path "${base_dir}" -type d)
