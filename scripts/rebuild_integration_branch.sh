#!/usr/bin/env bash
#
# Rebuild modernisation/integration by merging every in-flight
# modernisation feature branch into a fresh copy of main.
#
# Used during the AR7 modernisation period to keep a single branch that
# contains all in-flight changes, for local testing of the combined
# feature set. Individual feature branches remain independent for
# review on their own PRs.
#
# This branch is NOT a PR target. Do not merge it to main; merge the
# individual feature branches via their PRs. See ARCHITECTURE_NOTES.md
# for context.
#
# Usage:
#   scripts/rebuild_integration_branch.sh         rebuild locally only
#   scripts/rebuild_integration_branch.sh --push  also force-push to origin
#
# Maintenance:
#   When a new modernisation/* branch is added (or one merges to main
#   and is deleted), update the BRANCHES list below.

set -euo pipefail

# Order matters only when branches depend on each other. Today only
# netcdf-writer depends on parallel-models, and netcdf-writer's branch
# already contains parallel-models' commits because it was stacked,
# so listing netcdf-writer alone is enough.
BRANCHES=(
  modernisation/architecture-notes
  modernisation/python-3.12
  modernisation/worker-counts
  modernisation/netcdf-writer
  modernisation/fair2-adapter
  modernisation/ciceroscmpy2-adapter
  modernisation/scmdata-pandas3
  modernisation/iamc-loader
  modernisation/cross-model-notebooks
)

INTEGRATION_BRANCH="modernisation/integration"
PUSH=0

if [[ "${1:-}" == "--push" ]]; then
  PUSH=1
elif [[ -n "${1:-}" ]]; then
  echo "ERROR: unknown argument '$1'. Use --push or nothing." >&2
  exit 2
fi

# Only block on uncommitted changes to tracked files; untracked files
# (local venvs, scratch data, etc.) are unaffected by branch switching
# and git reset --hard, so they are safe.
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: tracked files have uncommitted changes. Commit or stash before running." >&2
  exit 1
fi

echo "==> Fetching from origin"
git fetch origin

echo "==> Resetting $INTEGRATION_BRANCH to origin/main"
if git show-ref --verify --quiet "refs/heads/$INTEGRATION_BRANCH"; then
  git checkout "$INTEGRATION_BRANCH"
else
  git checkout -b "$INTEGRATION_BRANCH"
fi
git reset --hard origin/main

for branch in "${BRANCHES[@]}"; do
  echo "==> Merging origin/$branch"
  git merge --no-ff "origin/$branch" -m "Merge $branch into integration"
done

if [[ $PUSH -eq 1 ]]; then
  echo "==> Force-pushing to origin (with-lease)"
  git push --force-with-lease origin "$INTEGRATION_BRANCH"
else
  echo "==> Skipping push (pass --push to push)"
fi

echo
echo "==> Done. Integration branch contains the following PR merges:"
git log --merges --oneline origin/main..HEAD
