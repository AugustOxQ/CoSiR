#!/usr/bin/env bash
# Resumable downloads for CoSiR evaluation benchmarks. Usage: download_benchmarks.sh {cub|semart|vg|all}
# Targets live on /data/SSD. Reruns resume (wget -c) and skip finished extractions.
set -euo pipefail
ROOT=/data/SSD
GDOWN=/root/miniconda3/bin/gdown
log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { log "FAIL: $*"; exit 1; }
fetch() { # url dest
  log "wget -c $1"; mkdir -p "$(dirname "$2")"
  wget -c --tries=20 --waitretry=10 --read-timeout=60 -O "$2" "$1" || die "download $1"
}
check_count() { # label found expected
  log "count $1: found=$2 expected=$3"; [ "$2" -eq "$3" ] || die "$1 count mismatch"
}

cub() {
  local d=$ROOT/cub; mkdir -p "$d/captions"
  # CUB-200-2011, CaltechDATA record 65de6-vp158 (1.2 GB, CC-BY, md5 97eceeb196236b17998738112f37df78)
  fetch "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1" "$d/CUB_200_2011.tgz"
  echo "97eceeb196236b17998738112f37df78  $d/CUB_200_2011.tgz" | md5sum -c - || die "CUB md5"
  [ -d "$d/CUB_200_2011/images" ] || { log "extract CUB"; tar -xzf "$d/CUB_200_2011.tgz" -C "$d"; }
  check_count "CUB images" "$(find "$d/CUB_200_2011/images" -name '*.jpg' | wc -l)" 11788
  # Reed et al. CVPR 2016 captions (10 per image). Drive file id from github.com/reedscot/cvpr2016.
  # MANUAL FALLBACK if gdown hits a quota or dead link: download the file in a browser from
  # https://drive.google.com/open?id=0B0ywwgffWnLLZW9uVHNjb2JmNlE and place it in $d/captions/
  local c=$d/captions
  if ! ls "$c"/*.zip "$c"/*.tar* "$c"/*.tgz >/dev/null 2>&1; then
    log "gdown reed captions"
    (cd "$c" && "$GDOWN" --continue "https://drive.google.com/uc?id=0B0ywwgffWnLLZW9uVHNjb2JmNlE") \
      || die "gdown failed (quota/dead link); see MANUAL FALLBACK in script"
  fi
  if [ ! -d "$c/extracted" ]; then
    log "extract captions"; mkdir -p "$c/extracted"
    for f in "$c"/*.zip; do [ -e "$f" ] && unzip -qn "$f" -d "$c/extracted"; done
    for f in "$c"/*.tar* "$c"/*.tgz; do [ -e "$f" ] && tar -xf "$f" -C "$c/extracted"; done
    # nested archives
    for f in $(find "$c/extracted" -maxdepth 2 \( -name '*.tar*' -o -name '*.tgz' \)); do tar -xf "$f" -C "$c/extracted"; done
  fi
  check_count "CUB caption txt (text_c10)" "$(find "$c/extracted" -path '*text_c10*' -name '*.txt' | wc -l)" 11788
}

semart() {
  local d=$ROOT/semart; mkdir -p "$d"
  # Aston Data Explorer eprint 380, CC BY-NC, ~3 GB, non-commercial research only
  fetch "https://researchdata.aston.ac.uk/id/eprint/380/1/SemArt.zip" "$d/SemArt.zip"
  unzip -tq "$d/SemArt.zip" >/dev/null || die "SemArt zip corrupt (rerun to resume)"
  [ -d "$d/SemArt" ] || { log "extract SemArt"; unzip -qn "$d/SemArt.zip" -d "$d"; }
  # paper (arXiv 1810.09617): 21,384 paintings
  check_count "SemArt images" "$(find "$d" -iname '*.jpg' | wc -l)" 21384
}

vg() {
  local d=$ROOT/visual_genome; mkdir -p "$d"
  # Stanford VG 1.2: images.zip -> VG_100K (64,346), images2.zip -> VG_100K_2 (43,731)
  fetch "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip"  "$d/images.zip"
  fetch "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip" "$d/images2.zip"
  for z in images images2; do unzip -tq "$d/$z.zip" >/dev/null || die "$z.zip corrupt"; done
  [ -d "$d/VG_100K" ]   || { log "extract images.zip";  unzip -qn "$d/images.zip"  -d "$d"; }
  [ -d "$d/VG_100K_2" ] || { log "extract images2.zip"; unzip -qn "$d/images2.zip" -d "$d"; }
  # GeneCIS config expects one flat dir of <image_id>.jpg ("VG_100K_all"); symlinks avoid duplicating the files
  mkdir -p "$d/VG_100K_all"
  find "$d/VG_100K" "$d/VG_100K_2" -name '*.jpg' -exec ln -sfn {} "$d/VG_100K_all/" \;
  check_count "VG images" "$(find "$d/VG_100K_all" -name '*.jpg' | wc -l)" 108077
  log "set /project/genecis/config.py visual_genome_images = $d/VG_100K_all (image_data.json is not needed by VAWDataset)"
}

case "${1:-}" in
  cub) cub;; semart) semart;; vg) vg;; all) cub; semart; vg;;
  *) echo "usage: $0 {cub|semart|vg|all}"; exit 2;;
esac
log "done: $1"
