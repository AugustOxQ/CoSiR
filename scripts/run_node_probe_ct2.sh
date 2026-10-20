#!/usr/bin/env bash
# r10 run 8: node probe for the claim test's stage 2. Can unchanged code see the local absolute paths
# (/data/..., /project/CoSiR/...) on this node, through a user namespace with bind mounts and a chroot?
# Prints facts only; writes only an empty folder inside the job's own code copy.
set -u
echo "== host"; hostname; sed -n 1,2p /etc/os-release; ldd --version | head -1; uname -r
echo "== gpu"; nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.limit --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} PWD=$PWD"
echo "== tools"
for t in unshare chroot apptainer singularity proot bwrap rsync flock; do printf '%s: %s\n' "$t" "$(command -v $t || echo none)"; done
echo "max_user_namespaces=$(cat /proc/sys/user/max_user_namespaces 2>/dev/null || echo unknown)"
ls -ld /data /project 2>&1
echo "== node data"
ls /local/wding /local/wding/Dataset /local/wding/Dataset/pre_extract /local/wding/pre_extract 2>&1 | head -60
ls -la /var/scratch/wding/Dataset/artelingo /local/wding/Dataset/artelingo 2>&1 | head -12
df -h /local/wding /var/scratch/wding 2>&1 | tail -2
echo "== userns"
unshare --user --map-root-user --mount true && echo "userns+mount: ok" || echo "userns+mount: FAIL"
echo "== fake root"
R=$(mktemp -d -p "$PWD" probe_root.XXXX)
PY=/var/scratch/wding/conda-envs/CoSiR/bin/python
unshare --user --map-root-user --mount bash -s "$R" "$PWD" "$PY" <<'EOS'
set -u; R=$1; CODE=$2; PY=$3
mount -t tmpfs tmpfs "$R" || { echo "tmpfs: FAIL"; exit 1; }
for d in bin sbin lib lib64 lib32 usr etc var home local opt dev proc sys run tmp srv cm; do
  [ -e "/$d" ] || continue
  if [ -L "/$d" ]; then ln -s "$(readlink "/$d")" "$R/$d"
  else mkdir -p "$R/$d" && mount --rbind "/$d" "$R/$d" || echo "bind /$d: FAIL"; fi
done
mkdir -p "$R/data/SSD2/pre_extract" "$R/project/CoSiR" &&
  mount --bind /local/wding/Dataset/pre_extract "$R/data/SSD2/pre_extract" &&
  mount --bind "$CODE" "$R/project/CoSiR" && echo "binds: ok"
chroot "$R" /bin/bash -c "ls /data/SSD2/pre_extract | head -5; ls /project/CoSiR | head -3; hostname; nvidia-smi -L;
  cd /project/CoSiR && $PY -c 'import torch; ok = torch.cuda.is_available(); print(\"torch\", torch.__version__, ok, torch.cuda.get_device_name(0) if ok else \"\"); x = torch.randn(512, 512, device=\"cuda\"); print(\"matmul ok\", bool(torch.isfinite(x @ x).all()))'" \
  && echo "chroot: ok" || echo "chroot: FAIL"
EOS
echo "== done"
