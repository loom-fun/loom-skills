#!/bin/bash
# Usage: new_campaign.sh CAMPAIGN_DIR NAME [INTEGRATION_BRANCH] [BASE_BRANCH]
# Creates the campaign state skeleton (see reference/campaign.md): BRIEF.md, DECISIONS.md, TECHLEAD.md, ENG_CONTRACT.md (notes/),
# status.json, log.md, inbox/{techlead,pm,dashboard}.md, notes/, eng/, wt/, and copies waitmail.sh and status.py into the directory.
# Placeholders in the templates are filled from the arguments; the rest are marked TODO for the tech lead.
set -eu
dir=${1:?usage: new_campaign.sh CAMPAIGN_DIR NAME [INTEGRATION_BRANCH] [BASE_BRANCH]}
name=${2:?campaign name}
integ=${3:-$name}
base=${4:-main}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tpl="$here/../../templates"
[ -e "$dir/status.json" ] && { echo "campaign exists: $dir" >&2; exit 1; }
mkdir -p "$dir"/{inbox,notes,eng,wt}
dir=$(cd "$dir" && pwd)
fill() {
  sed -e "s|@CAMPAIGN@|$name|g" -e "s|@DIR@|$dir|g" -e "s|@INTEGRATION@|$integ|g" -e "s|@BASE@|$base|g" \
      -e "s|@DATE@|$(date +%F)|g" "$1"
}
fill "$tpl/BRIEF.md" > "$dir/BRIEF.md"
fill "$tpl/ENG_CONTRACT.md" > "$dir/notes/ENG_CONTRACT.md"
fill "$tpl/task.md" > "$dir/notes/task_TEMPLATE.md"
printf '# Decisions (design decisions and deviations from the plan, each with the reason)\n\nBase: %s. Deviations are marked DEV.\n' "$base" > "$dir/DECISIONS.md"
printf '# TECHLEAD state (read first after a restart or compaction)\n- Brief: BRIEF.md. Decisions: DECISIONS.md. Engineers: notes/task_<id>.md.\n- Integration branch: %s (from %s). Worktrees: wt/<id> on %s-<id>.\n## Progress\n' "$integ" "$base" "$integ" > "$dir/TECHLEAD.md"
cat > "$dir/status.json" <<JSON
{"workstreams": {}, "models": {}, "measures": {}, "velocity": {"stages": {}}}
JSON
: > "$dir/log.md"
for m in techlead pm dashboard; do : > "$dir/inbox/$m.md"; done
cp "$here/waitmail.sh" "$here/status.py" "$dir/"
python3 "$here/status.py" --dir "$dir" log tech-lead "campaign $name created (integration branch $integ, base $base)"
echo "created $dir; next: edit BRIEF.md (goal, measures, deliverables), then write notes/task_<id>.md per engineer"
