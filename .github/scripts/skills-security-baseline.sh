#!/usr/bin/env bash
set -euo pipefail

repository=https://github.com/loom-fun/loom-skills.git
evidence=/evidence
mkdir -p /tmp/install-project
cd /tmp/install-project
git ls-remote "$repository" HEAD > "$evidence/repository-before.txt"
node --version > "$evidence/node-version.txt"
npx skills --version | tee "$evidence/skills-version.txt"

terminal="$evidence/public-terminal.log"
coproc PUBLIC_INSTALL {
  script -qefc "npx skills add loom-fun/loom-skills --skill '*'" "$terminal" >/dev/null
}
installer_pid="$PUBLIC_INSTALL_PID"
exec {installer_input}>&"${PUBLIC_INSTALL[1]}"

declare -A answered=()
prompts=(
  'Which agents do you want to install to?'
  'Installation scope'
  'Installation method'
  'Proceed with installation?'
)
for (( second=0; second<180; second++ )); do
  if ! kill -0 "$installer_pid" 2>/dev/null; then
    break
  fi
  if [[ -f "$terminal" ]]; then
    for prompt in "${prompts[@]}"; do
      if [[ ! ${answered[$prompt]+seen} ]] && grep -aFq "$prompt" "$terminal"; then
        printf '\r' >&"$installer_input"
        answered[$prompt]=yes
      fi
    done
    if grep -aFq 'Install the' "$terminal" && grep -aFq 'find-skills' "$terminal"; then
      printf '\003' >&"$installer_input"
      break
    fi
  fi
  sleep 1
done
if kill -0 "$installer_pid" 2>/dev/null && [[ ! ${answered['Proceed with installation?']+seen} ]]; then
  printf '\003' >&"$installer_input"
fi
exec {installer_input}>&-
wait "$installer_pid"

git ls-remote "$repository" HEAD > "$evidence/repository-after.txt"
cmp "$evidence/repository-before.txt" "$evidence/repository-after.txt"
node --input-type=module - "$terminal" "$evidence" <<'JS'
import { createHash } from 'node:crypto';
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const [, , terminalPath, evidence] = process.argv;
const log = readFileSync(terminalPath, 'utf8')
  .replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '')
  .replace(/\r/g, '');
writeFileSync(join(evidence, 'public-install.txt'), log);

const skillRoot = '.agents/skills';
const skills = readdirSync(skillRoot).sort();
if (!skills.includes('loom') || skills.length < 10) {
  throw new Error('The public interactive installation yielded too few skills');
}
const installed = [];
function record(directory) {
  for (const file of readdirSync(directory, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
    const path = join(directory, file.name);
    if (file.isDirectory()) record(path);
    else if (file.isFile()) installed.push({ path, sha256: createHash('sha256').update(readFileSync(path)).digest('hex') });
  }
}
record(skillRoot);
writeFileSync(join(evidence, 'installed-files.json'), JSON.stringify({ skills, files: installed }, null, 2));
const audit = new URL('https://www.skills.sh/tele/audit');
audit.search = new URLSearchParams({ source: 'loom-fun/loom-skills', skills: skills.join(',') });
const response = await fetch(audit, { signal: AbortSignal.timeout(30000) });
if (!response.ok) throw new Error(`Public audit API returned ${response.status}`);
writeFileSync(join(evidence, 'provider-assessments.json'), JSON.stringify(await response.json(), null, 2));
if (!log.includes('Security Risk Assessments')) {
  throw new Error('The CLI did not display partner assessments; the public API output was retained');
}
JS
