#!/usr/bin/env bash
set -euo pipefail
cd /gpfs/home1/paul6598/AAMAS
export SC2PATH=/gpfs/home1/paul6598/StarCraftII
exec timeout --signal=TERM --kill-after=60s 24h /home1/paul6598/miniconda3/envs/aamas/bin/python main.py --config=rsvp --env-config=sc2 with results/diagnostics/rsvp_fixed200_20260907/config.json seed=1
