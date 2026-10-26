#!/usr/bin/env bash
# Point git at the repo's tracked hooks. Run once after cloning.
#
# .git/hooks is not version controlled, so hooks live in .githooks/ and
# core.hooksPath points there -- which means a fresh clone gets them with one
# command instead of a wiki page nobody reads.
set -euo pipefail
cd "$(dirname "$0")/.."
chmod +x .githooks/*
git config core.hooksPath .githooks
echo "Hooks installed: $(ls .githooks | tr '\n' ' ')"
