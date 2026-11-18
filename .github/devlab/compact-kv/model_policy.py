#!/usr/bin/env python3
"""Matched one-session/no-cache policy and source-only unsupported-build guards."""
import argparse
import json
from pathlib import Path

POLICY=dict(max_sessions=1,session_cache_capacity=0,prefix_cache_capacity=0,cache_target_bytes=0,checkpoint_capacity=0)


def config(src,dst,arm):
    d=json.loads(Path(src).read_text())
    d.setdefault('choices',{}).update({'memory.kv.compact_gemma4_e4b':arm=='candidate','target.rdna4_twins':False})
    Path(dst).write_text(json.dumps(d,indent=1)+'\n')


def card(package,arm):
    package=Path(package);path=package/'model.json';d=json.loads(path.read_text())
    if set(d['presets']) != {'chat'}:raise ValueError('expected exact target-only chat preset')
    e=d['presets']['chat']['execution']
    if any(e.get(k)!=v for k,v in dict(rows=1,context_capacity=65536,prefill_capacity=1024).items()):
        raise ValueError('both arms must use rows1/context65536/prefill1024')
    if arm=='candidate' and any(e.get(k)!=v for k,v in POLICY.items()):
        raise ValueError('candidate did not emit the supported cache policy')
    before=dict(e);e.update(POLICY)
    (package.parent/(package.name+'.matched_policy.json')).write_text(json.dumps({'arm':arm,'before':before,'after':e},indent=2)+'\n')
    path.write_text(json.dumps(d,indent=1)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    c=s.add_parser('config');c.add_argument('source');c.add_argument('output');c.add_argument('arm',choices=['baseline','candidate'])
    c=s.add_parser('card');c.add_argument('package');c.add_argument('arm',choices=['baseline','candidate'])
    a=p.parse_args()
    if a.command=='config':config(a.source,a.output,a.arm)
    else:card(a.package,a.arm)
