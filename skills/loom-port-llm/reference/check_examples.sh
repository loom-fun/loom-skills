#!/bin/bash
# usage: check_examples.sh   - every reference example still applies to the current starter and compiles (CI)
set -u
here=$(cd "$(dirname "$0")" && pwd); tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT; bad=0
for ex in "$here"/*-example; do
  n=$(basename "$ex")
  if out=$("$here/apply_example.sh" "$n" "$tmp/$n" 2>&1) && find "$tmp/$n" -name "*.py" -print0 | xargs -0 python3 -m py_compile; then
    echo "OK   $n"
  else echo "FAIL $n: $out"; bad=1; fi
done
exit $bad
