#!/usr/bin/env bash
# Fetch the public model from its original source, retaining its licence.
set -euo pipefail
demo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="$demo_root/vendor/rebotarm_digital_twin_dm"
model_commit=914c763c4adffb5f5744c43ecffbf6199c282bd6
if [[ -e "$model_dir" ]]; then
  printf 'Model directory already exists; left unchanged: %s\n' "$model_dir"
  exit 1
fi
mkdir -p "$demo_root/vendor"
git clone --filter=blob:none --no-checkout https://github.com/Yang-Ci/ReBot_Arm_DigitalTwin_DM.git "$model_dir"
git -C "$model_dir" checkout --detach "$model_commit"
printf 'Model checked out at %s. Read its CERN-OHL-W-2.0 LICENSE before redistribution.\n' "$model_commit"
