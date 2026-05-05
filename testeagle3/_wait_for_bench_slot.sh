#!/bin/bash
# Sourced by bench_*.sh. Blocks until one of MAX_SLOTS concurrent slots
# is free on this node, then holds it for the rest of the script's life
# via an open file descriptor (released automatically on exit).

MAX_SLOTS="${MAX_SLOTS:-3}"
LOCK_ROOT="/data/ai_club/smeargle/testeagle3/.bench_slots"
LOCKDIR="${LOCK_ROOT}/$(hostname)"
mkdir -p "$LOCKDIR"

_SLOT_FD=""
_SLOT_NUM=""
_wait_start=$(date +%s)

while [[ -z "$_SLOT_FD" ]]; do
  for i in $(seq 1 "$MAX_SLOTS"); do
    exec {fd}>"$LOCKDIR/slot_$i.lock"
    if flock -n "$fd"; then
      _SLOT_FD=$fd
      _SLOT_NUM=$i
      break
    fi
    exec {fd}>&-
  done
  if [[ -z "$_SLOT_FD" ]]; then
    waited=$(( $(date +%s) - _wait_start ))
    echo "[slot] all $MAX_SLOTS slots on $(hostname) busy (waited ${waited}s, job ${SLURM_JOB_ID:-?}); sleeping 30s"
    sleep 30
  fi
done

echo "[slot] acquired slot $_SLOT_NUM on $(hostname) after $(( $(date +%s) - _wait_start ))s (job ${SLURM_JOB_ID:-?})"
