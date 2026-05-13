#!/bin/bash
set -euo pipefail

USER_NAME=${1:-${USER:-$(whoami)}}
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

USER_JOBS="${TMP_DIR}/user_jobs"
POSITIONS="${TMP_DIR}/positions"
: > "$POSITIONS"

squeue -u "$USER_NAME" -h -o "%i|%P|%j|%T|%M|%L|%D|%R" > "$USER_JOBS"

if [ ! -s "$USER_JOBS" ]; then
  echo "No jobs found for user ${USER_NAME}."
  exit 0
fi

cut -d'|' -f2 "$USER_JOBS" | sort -u | while IFS= read -r partition; do
  [ -n "$partition" ] || continue
  pending_ids="${TMP_DIR}/pending_${partition//[^A-Za-z0-9_.-]/_}"
  squeue -p "$partition" -t PD -h -o "%i" > "$pending_ids"
  total=$(awk 'END {print NR + 0}' "$pending_ids")
  awk -v total="$total" '{print $1 "|" NR "|" total "|" (NR - 1)}' "$pending_ids" >> "$POSITIONS"
done

printf "%-18s %-9s %-32s %-10s %-11s %-11s %-5s %-11s %-7s %s\n" \
  JOBID PARTITION NAME STATE TIME TIME_LEFT NODES POS_TOTAL AHEAD REASON

awk -F'|' '
  NR == FNR {
    pos[$1] = $2 "/" $3
    ahead[$1] = $4
    next
  }
  {
    pos_total = "-"
    ahead_count = "-"
    if ($4 == "PENDING") {
      pos_total = (($1 in pos) ? pos[$1] : "?")
      ahead_count = (($1 in ahead) ? ahead[$1] : "?")
    }
    printf "%-18s %-9s %-32s %-10s %-11s %-11s %-5s %-11s %-7s %s\n",
      $1, $2, substr($3, 1, 32), $4, $5, $6, $7, pos_total, ahead_count, $8
  }
' "$POSITIONS" "$USER_JOBS"
