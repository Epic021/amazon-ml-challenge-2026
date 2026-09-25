#!/usr/bin/env bash
# One-time setup for the team VM (Ubuntu 22.04). Safe to re-run.
#   curl -fsSL https://raw.githubusercontent.com/Epic021/amazon-ml-challenge-2026/main/scripts/setup_vm.sh | bash
set -euo pipefail

REPO_URL=https://github.com/Epic021/amazon-ml-challenge-2026.git
WORK=/work
REPO=$WORK/amazon-ml-challenge-2026

sudo apt-get update -y
sudo apt-get install -y python3-venv python3-dev build-essential git tmux htop unzip

# Shared folder that every teammate's user can write to
sudo mkdir -p "$WORK" && sudo chmod 777 "$WORK"

[ -d "$REPO/.git" ] || git clone "$REPO_URL" "$REPO"
git config --global --add safe.directory "$REPO"

# One shared virtualenv for the team
[ -d "$WORK/venv" ] || python3 -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install --upgrade pip wheel
"$WORK/venv/bin/pip" install -r "$REPO/requirements.txt"

# Auto-activate the venv and cd into the repo on login
grep -q "/work/venv/bin/activate" ~/.bashrc || cat >> ~/.bashrc <<'EOF'
source /work/venv/bin/activate
cd /work/amazon-ml-challenge-2026
EOF

echo
echo "Setup done. Next:"
echo "  1. Put the dataset at $REPO/student_resource/dataset/{train,test}/*.tsv"
echo "  2. python scripts/tsv_to_parquet.py"
