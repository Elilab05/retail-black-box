#!/usr/bin/env bash
# Local helper; NEVER called by the data pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v gh >/dev/null 2>&1; then
  echo "Install GitHub CLI first: https://cli.github.com/" >&2
  exit 1
fi
if ! gh auth status; then
  echo "Authenticate with 'gh auth login' before publishing." >&2
  exit 1
fi
printf 'Create PUBLIC GitHub repo retail-black-box (raw challenge CSVs excluded)? Type PUBLICAR: '
read -r answer
if [ "$answer" != "PUBLICAR" ]; then
  echo 'Publication canceled.'
  exit 1
fi
if [ ! -d .git ]; then git init -b main; fi
git add .
if ! git diff --cached --quiet; then
  git commit -m "Initial Retail Black Box proof of concept"
fi
gh repo create retail-black-box --public --source=. --remote=origin --push
