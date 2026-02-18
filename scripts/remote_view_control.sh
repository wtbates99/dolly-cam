#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <status|enable|disable> <base_url> <access_token> <admin_token>"
  echo "Example: $0 disable http://192.168.1.50:8080 ACCESS_TOKEN ADMIN_TOKEN"
  exit 1
fi

ACTION="$1"
BASE_URL="${2%/}"
ACCESS_TOKEN="$3"
ADMIN_TOKEN="$4"

case "$ACTION" in
  status)
    curl -fsS "${BASE_URL}/api/remote-view?token=${ACCESS_TOKEN}" ; echo
    ;;
  disable|enable)
    curl -fsS -X POST \
      "${BASE_URL}/api/remote-view?action=${ACTION}&token=${ACCESS_TOKEN}&admin_token=${ADMIN_TOKEN}" ; echo
    ;;
  *)
    echo "Invalid action: $ACTION"
    exit 1
    ;;
esac
