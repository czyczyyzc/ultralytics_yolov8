#!/bin/bash
set -euo pipefail

set_governor() {
  local path="$1"
  local value="$2"
  if [ -w "$path" ]; then
    echo "$value" >"$path"
  fi
}

for path in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
  [ -e "$path" ] || continue
  set_governor "$path" performance
done

for path in /sys/class/devfreq/dmc/governor /sys/class/devfreq/*npu/governor; do
  [ -e "$path" ] || continue
  set_governor "$path" performance
done

echo "RK3588 governors pinned to performance"
