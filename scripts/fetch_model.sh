#!/usr/bin/env bash
# Fetch pinned geometry sources, retaining their distinct provenance.
set -euo pipefail
demo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$demo_root/vendor"
fetch_pinned() {
  local source_url="$1" source_dir="$2" source_commit="$3"
  if [[ -e "$source_dir" ]]; then
    if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$source_commit" ]]; then
      printf 'Existing checkout differs from pinned revision; left unchanged: %s\n' "$source_dir" >&2
      return 1
    fi
    printf 'Pinned checkout already present: %s\n' "$source_dir"
    return
  fi
  git clone --filter=blob:none --no-checkout "$source_url" "$source_dir"
  git -C "$source_dir" checkout --detach "$source_commit"
}
fetch_pinned https://github.com/Yang-Ci/ReBot_Arm_DigitalTwin_DM.git \
  "$demo_root/vendor/rebotarm_digital_twin_dm" 914c763c4adffb5f5744c43ecffbf6199c282bd6
fetch_pinned https://github.com/Seeed-Projects/reBotArmController_ROS2.git \
  "$demo_root/vendor/rebotarm_ros2" 4a2b31b569d7856ea310b90f24f7de82374dbc75
printf 'Read docs/SOURCE_PROVENANCE.md and retained CERN-OHL-W-2.0 notices before redistributing model-derived geometry.\n'
