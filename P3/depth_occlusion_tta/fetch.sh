#!/usr/bin/env bash
set -eu
: "${OCC_DATA:?Set OCC_DATA to a data directory on Asus}"
mkdir -p "$OCC_DATA/ibims" "$OCC_DATA/midd"
# Public access credentials are published in the MediaTUM dataset record.
export RSYNC_PASSWORD=m1455541
server=rsync://m1455541@dataserv.ub.tum.de/m1455541
for f in Ibims_Dataset.sha512 imagelist.txt ibims1_core_raw.zip ibims1_core_mat.zip evaluation_scripts.zip ibims1_aux_mvs.zip ibims1_aux_augmented.zip ibims1_aux_patterns.zip; do
  rsync -t --partial "$server/$f" "$OCC_DATA/ibims/"
done
(cd "$OCC_DATA/ibims"; tr -d '\r' < Ibims_Dataset.sha512 | sha512sum -c -)
url=https://vision.middlebury.edu/stereo/data/scenes2014
curl -fsSL "$url/" | grep -oE 'datasets/[A-Za-z0-9]+-perfect' | sort -u > "$OCC_DATA/midd/list.txt"
while read -r folder; do
  scene=$(basename "$folder" -perfect)
  mkdir -p "$OCC_DATA/midd/$scene"
  for file in calib.txt im0.png disp0.pfm; do
    p="$OCC_DATA/midd/$scene/$file"
    if [ ! -s "$p" ]; then
      curl -fLsS --retry 4 "$url/$folder/$file" -o "$p.tmp" && mv "$p.tmp" "$p"
    fi
  done
done < "$OCC_DATA/midd/list.txt"
