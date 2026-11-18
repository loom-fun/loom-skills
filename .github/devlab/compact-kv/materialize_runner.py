#!/usr/bin/env python3
"""Derive a reviewable compact job from the exact already-reviewed ordinary runner.

CPU/source only. No checkpoint access, network, model launch or GPU execution.
Every substitution is a single exact anchor. Derived diffs and source hashes are
saved, and the original monitoring/readiness/settle implementation is retained.
"""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
PIN='7abe4277c4ff30195592f80ff170df1b3ad19985'
PREFIX='eng/gemma-e4b-min-ram/'
FILES=['run_job.sh','run_frontier.sh','prepare_packages.sh','cold_capacity_targeted.py',
       'compare_arms.py','runner/sdk_pin.py','target_build_config.json','target_text_llm_config.json']


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'expected one frozen source anchor: {old!r}')
    return text.replace(old,new,1)


def patch_sources(source, head, kit_sha, builder_sha):
    result=dict(source)
    p=source['prepare_packages.sh']
    p=replace_once(p,'ROOT=$(git rev-parse --show-toplevel)','ROOT=${LOOM_COMPACT_CHECKOUT:?}')
    p=replace_once(p,'EXPECTED_KIT=e40fd8d3d77174bf809860ae91f6db2214a9e207dad341b663dc3af22f0127fd','EXPECTED_KIT='+kit_sha)
    p=replace_once(p,'EXPECTED_BUILDER=faeb722cee49d011cd3422d79832dfb9533710845f7f9e8041b841ff9d363db4','EXPECTED_BUILDER='+builder_sha)
    p=replace_once(p,"source['skills_commit'] != 'bacdd3ed241eb62f9578b909ef63eaa0751f3c49'", "source['skills_commit'] != '"+head+"'")
    p=replace_once(p," or source.get('kb_root') is not None",'')
    p=replace_once(p,'if [ "$arm" = baseline ]; then context=131072; rows=4; else context=65536; rows=1; fi',
        '''context=65536; rows=1
  "$BUILD_PY" "$ROOT/.github/devlab/compact-kv/model_policy.py" config \\
    "$OUT/target_text_build_config.runtime.json" "$OUT/$arm.config.json" "$arm"''')
    p=replace_once(p,'"$builder" "$OUT/target_text_build_config.runtime.json" "$pkg"', '"$builder" "$OUT/$arm.config.json" "$pkg"')
    p=replace_once(p,'--prefill 1024 --memory-gib auto --draft none', '--pool 65536 --prefill 1024 --memory-gib auto --draft none')
    p=replace_once(p,'  "$BUILD_PY" - "$pkg/model.json" "$context" "$rows"',
        '''  "$BUILD_PY" "$ROOT/.github/devlab/compact-kv/model_policy.py" card "$pkg" "$arm"
  "$BUILD_PY" - "$pkg/model.json" "$context" "$rows"''')
    result['prepare_packages.sh']=p
    p=source['run_job.sh']
    p=replace_once(p,'ROOT=$(git -C "$HERE" rev-parse --show-toplevel)','ROOT=${LOOM_COMPACT_CHECKOUT:?}')
    p=replace_once(p,'  rm -f "$OUT/builds/target_text_build_config.runtime.json"',
        '  rm -f "$OUT/builds/target_text_build_config.runtime.json" "$OUT/builds/baseline.config.json" "$OUT/builds/candidate.config.json"\n  rm -f "$OUT/kernel-fixtures/"*.npy "$OUT/kernel-fixtures/"*.loom')
    result['run_job.sh']=p
    p=source['run_frontier.sh']
    p=replace_once(p,'ROOT=$(git rev-parse --show-toplevel)','ROOT=${LOOM_COMPACT_CHECKOUT:?}')
    p=replace_once(p,'--target-prompt-tokens 65535','--target-prompt-tokens 65536')
    p=replace_once(p,'BASELINE="$OUT/builds/baseline"', '''# Strict kernel gate before either model server starts. All data remain local.
ADAPTER="$ROOT/.github/devlab/compact-kv"
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 timeout --signal=TERM --kill-after=5s 60s \\
  "$VENV/bin/python" "$ADAPTER/generate_checks.py" \\
  --legacy "$OUT/builds/baseline/program/llm.loom" \\
  --candidate "$OUT/builds/candidate/program/llm.loom" --out "$OUT/kernel-fixtures"
timeout --signal=TERM --kill-after=5s 425s "$VENV/bin/python" "$ADAPTER/run_checks.py" \\
  --tool "$VENV/bin/loom-test" --fixtures "$OUT/kernel-fixtures" --out "$OUT/kernel-results" --budget 420
BASELINE="$OUT/builds/baseline"''')
    result['run_frontier.sh']=p
    p=source['cold_capacity_targeted.py']
    p=replace_once(p,"        result['phase'] = 'health_after'", '''        # Separate request after the full context: both cards disable all caches.
        # This reuses the SDK slot, with fresh positions starting at zero.
        result['phase'] = 'fresh_session_reset'
        code, body = request.call('reset_short', '/v1/chat/completions', payload(short))
        result['reset_usage'] = completed(code, body, short_tokens)
        result['reset_text'] = body['choices'][0]['message']['content']
        if result['reset_text'] != result['short_text']:
            raise ValueError('fresh-session short answer changed after full-context request')
        result['phase'] = 'health_after' ''')
    result['cold_capacity_targeted.py']=p
    p=source['compare_arms.py'].replace('65535','65536')
    p=replace_once(p,"('short_text', 'retrieval_text')", "('short_text', 'retrieval_text', 'reset_text')")
    result['compare_arms.py']=p
    return result


def materialize(out, expected_head):
    def git(*args):
        return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    head=git('rev-parse','HEAD')
    if head != expected_head or git('status','--porcelain','--untracked-files=no'):
        raise ValueError('source checkout must be the exact clean reviewed head')
    out.mkdir(parents=True,exist_ok=False)
    source={name:subprocess.check_output(['git','-C',str(ROOT),'show',PIN+':'+PREFIX+name]).decode() for name in FILES}
    subprocess.run(['bash',str(ROOT/'skills/loom-port-llm/reference/apply_example.sh'),'gemma4',str(out/'kit')],check=True)
    kit=json.loads((out/'kit/KIT_SOURCE').read_text())
    if kit['dirty'] is not False or kit['skills_commit'] != head:
        raise ValueError('materialized kit is not from the clean reviewed checkout')
    sha=lambda data:hashlib.sha256(data).hexdigest()
    result=patch_sources(source,head,kit['kit_sha256'],sha((out/'kit/build.py').read_bytes()))
    diff=[];records={}
    for name,text in result.items():
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
        if name.endswith('.sh'):path.chmod(0o755)
        records[name]={'ordinary_sha256':sha(source[name].encode()),'derived_sha256':sha(text.encode())}
        diff.extend(difflib.unified_diff(source[name].splitlines(True),text.splitlines(True),'ordinary/'+name,'compact/'+name))
    (out/'runner.diff').write_text(''.join(diff))
    (out/'runner-manifest.json').write_text(json.dumps({'ordinary_head':PIN,'compact_head':head,'kit':kit,'files':records},indent=2)+'\n')
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--expected-head',required=True)
    a=p.parse_args();materialize(a.out.resolve(),a.expected_head)
