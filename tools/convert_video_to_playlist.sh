#!/usr/bin/env bash

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_video> <output_dir>"
    exit 1
fi

input_video="$1"
output_dir="$2"

ffmpeg \
    -i "${input_video}" \
    -codec copy -map 0 -f segment \
    -segment_list "${output_dir}/playlist.m3u8" \
    -segment_list_flags +live \
    -segment_time 5 "${output_dir}/out%03d.ts"
