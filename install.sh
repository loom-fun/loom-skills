#!/bin/bash
# Reference install without Node: copy the skills into ~/.agents/skills (Codex, Cursor, Gemini CLI, ...)
# and link them into ~/.claude/skills, which Claude Code reads.
# usage: ./install.sh            (or: curl/clone the repo first)
#        ./install.sh --uninstall
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
dest=${AGENTS_SKILLS_DIR:-$HOME/.agents/skills}
claude=${CLAUDE_SKILLS_DIR:-$HOME/.claude/skills}
skills=(loom loom-setup loom-demo-ui loom-port-llm loom-roofline loom-optimize loom-benchmark loom-improve)
if [ "${1:-}" = "--uninstall" ]; then
  for s in "${skills[@]}"; do rm -rf "$dest/$s" "$claude/$s"; done; echo "removed ${skills[*]} from $dest and $claude"; exit 0
fi
mkdir -p "$dest" "$claude"
for s in "${skills[@]}"; do
  rm -rf "$dest/$s" "$claude/$s"; cp -a "$here/skills/$s" "$dest/$s"; ln -s "$dest/$s" "$claude/$s"
done
echo "installed ${#skills[@]} skills into $dest (linked from $claude) - start a new agent session and run /loom"
