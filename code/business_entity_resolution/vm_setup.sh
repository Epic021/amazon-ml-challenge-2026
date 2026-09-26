#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu VM. Expects ~/dataset.tar.zst and ~/ber/code/... (copied by scp).
set -euo pipefail
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip zstd htop >/dev/null
python3 -m venv ~/venv
. ~/venv/bin/activate
pip install -q --upgrade pip
pip install -q -r ~/ber/code/business_entity_resolution/requirements.txt
mkdir -p ~/ber/dataset/student_resource
tar -I zstd -xf ~/dataset.tar.zst -C ~/ber/dataset/student_resource
python3 -c "import lightgbm, numba, rapidfuzz, jellyfish, anyascii, pyarrow; print('python deps ok')"
nproc; free -g; df -h ~ | tail -1
