#!/usr/bin/env bash
set -euo pipefail

USER_FILTER=${1:-${SLURM_JOB_USER:-${USER:-$(whoami)}}}

if ! command -v squeue >/dev/null 2>&1; then
  echo "squeue is not available on this host. Run this on a Slurm login node."
  exit 1
fi

echo "Queue positions for user: ${USER_FILTER}"
echo "Checked: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo

mapfile -t JOBS < <(squeue -h -u "$USER_FILTER" -o "%i|%P|%j|%T|%Q|%M|%l|%D|%R" 2>/dev/null || true)

if [ "${#JOBS[@]}" -eq 0 ]; then
  echo "No queued or running jobs for ${USER_FILTER}."
  exit 0
fi

declare -A START_ESTIMATES
while IFS='|' read -r job_id start_time; do
  [ -n "${job_id:-}" ] || continue
  START_ESTIMATES["$job_id"]=${start_time:-N/A}
done < <(squeue --start -h -u "$USER_FILTER" -o "%i|%S" 2>/dev/null || true)

partition_queue() {
  local partition=$1
  squeue -h -p "$partition" -t PD --sort=-Q,i -o "%i|%u|%P|%j|%T|%Q|%D|%R" 2>/dev/null || true
}

pending_position() {
  local partition=$1
  local target_job=$2
  partition_queue "$partition" | awk -F'|' -v target="$target_job" '$1 == target { print NR; found=1; exit } END { if (!found) print "-" }'
}

pending_total() {
  local partition=$1
  partition_queue "$partition" | wc -l | tr -d ' '
}

printf "%-18s %-9s %-32s %-9s %-8s %-7s %-12s %-9s %-20s %s\n" \
  "JOBID" "PART" "NAME" "STATE" "PRIOR" "NODES" "TIME" "POS/TOTAL" "START_EST" "REASON/NODELIST"

for row in "${JOBS[@]}"; do
  IFS='|' read -r job_id partition name state priority elapsed limit nodes reason <<< "$row"

  position="-"
  start_estimate="${START_ESTIMATES[$job_id]:-N/A}"
  if [ "$state" = "PENDING" ]; then
    position=$(pending_position "$partition" "$job_id")
    if [ "$position" != "-" ]; then
      total=$(pending_total "$partition")
      position="${position}/${total}"
    fi
  fi

  printf "%-18s %-9s %-32.32s %-9s %-8s %-7s %-12s %-9s %-20s %s\n" \
    "$job_id" "$partition" "$name" "$state" "$priority" "$nodes" "${elapsed}/${limit}" "$position" "$start_estimate" "$reason"
done

echo
echo "POS/TOTAL is the raw pending position within that partition and total pending jobs there, sorted like:"
echo "  squeue -p <partition> -t PD --sort=-Q,i"
echo "Example: 7/16 means 7th pending in that partition out of 16 total pending jobs."
