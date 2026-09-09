#!/bin/bash
# Legacy entry point for queued commands. New experiments use run_rsvp_grf.sh.
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_rsvp_grf.sh" "$@"
