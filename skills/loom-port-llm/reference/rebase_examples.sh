#!/bin/bash
# usage: rebase_examples.sh OLD_STARTER [EXAMPLE...]   (maintainers: after changing ../starter)
# Each example's starter.patch is a diff against the starter. After a starter change, rebuild the
# patches: apply each patch to OLD_STARTER (a copy of the starter they were made against, e.g.
# `git worktree add /tmp/old HEAD` + /tmp/old/skills/loom-port-llm/starter), 3-way merge every patched
# file onto the current starter (git merge-file), and rewrite starter.patch. Conflicts are left in
# the printed work dir with <<<<<<< markers: resolve them there and rerun with OLD_STARTER=that dir's
# "merged" copy, or fix the patch by hand. check_examples.sh must pass afterwards.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); new=$here/../starter; old=$(cd "$1" && pwd); shift
exs=("$@"); [ ${#exs[@]} -gt 0 ] || exs=($(cd "$here" && ls -d *-example))
work=$(mktemp -d); bad=0
for e in "${exs[@]}"; do
  ex=$here/${e%-example}-example; n=$(basename "$ex")
  cp -r "$old" "$work/$n.old"; cp -r "$old" "$work/$n.full"
  patch -p1 -s --no-backup-if-mismatch -d "$work/$n.full" < "$ex/starter.patch"
  files=$(grep '^+++ b/' "$ex/starter.patch" | cut -c7- | sort -u)
  : > "$work/$n.patch"
  for f in $files; do
    mkdir -p "$(dirname "$work/$n.merged/$f")"; cp "$new/$f" "$work/$n.merged/$f"
    if ! git merge-file -q "$work/$n.merged/$f" "$work/$n.old/$f" "$work/$n.full/$f"; then
      echo "CONFLICT $n/$f: resolve $work/$n.merged/$f"; bad=1
    fi
    diff -u --label "a/$f" --label "b/$f" "$new/$f" "$work/$n.merged/$f" >> "$work/$n.patch" || true
  done
  [ $bad = 1 ] || { cp "$work/$n.patch" "$ex/starter.patch"; echo "rebased $n ($(wc -l < "$ex/starter.patch") lines)"; }
done
[ $bad = 1 ] && { echo "work dir: $work"; exit 1; }
rm -rf "$work"; "$here/check_examples.sh"
