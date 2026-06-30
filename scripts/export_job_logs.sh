#!/bin/bash
set -euo pipefail

usage() {
  echo "Usage: $0 JOBID [--prefix GLOB_PREFIX]"
  echo
  echo "Examples:"
  echo "  $0 793687"
  echo "  $0 793687 --prefix eval-sab-react-30b-inst-smoke"
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ $# -lt 1 ]; then
  usage
  exit 0
fi

JOBID="$1"
shift || true

PREFIX="*"
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix)
      PREFIX="${2:?missing value for --prefix}"
      shift 2
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
LOG_DIR=${LOG_DIR:-$PROJECT_ROOT/logs}
EXPORT_ROOT=${EXPORT_ROOT:-$PROJECT_ROOT/log_exports}
DEST_DIR="$EXPORT_ROOT/job-$JOBID"
ARCHIVE="$EXPORT_ROOT/job-$JOBID.tgz"

mkdir -p "$DEST_DIR"

shopt -s nullglob
matches=(
  "$LOG_DIR"/${PREFIX}.${JOBID}.*
  "$LOG_DIR"/${PREFIX}.${JOBID}*
)
shopt -u nullglob

if [ ${#matches[@]} -eq 0 ]; then
  echo "ERROR: no log files found for job $JOBID under $LOG_DIR" >&2
  echo "Hint: ls -lh $LOG_DIR/*$JOBID*" >&2
  exit 1
fi

echo "Exporting ${#matches[@]} log file(s) for job $JOBID"
rm -f "$DEST_DIR"/*
for path in "${matches[@]}"; do
  if [ -f "$path" ]; then
    cp -p "$path" "$DEST_DIR/"
    ls -lh "$path"
  fi
done

tar -czf "$ARCHIVE" -C "$EXPORT_ROOT" "job-$JOBID"
ls -lh "$ARCHIVE"
echo "Done: $ARCHIVE"
