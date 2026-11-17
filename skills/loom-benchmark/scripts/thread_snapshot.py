#!/usr/bin/env python3
"""Bounded, passive Linux process/thread evidence; diagnostic only, no GPU API."""
import argparse
import json
import os
from pathlib import Path
import time


def read_text(path):
    try:
        return path.read_text(), None
    except (OSError, ValueError) as exc:
        return None, type(exc).__name__


def status_fields(text):
    if text is None:
        return None
    keep = {'Name', 'State', 'Threads', 'Cpus_allowed_list', 'Mems_allowed_list',
            'voluntary_ctxt_switches', 'nonvoluntary_ctxt_switches'}
    return {key: value.strip() for line in text.splitlines() if ':' in line
            for key, value in [line.split(':', 1)] if key in keep}


def stat_fields(text):
    if text is None:
        return None
    fields = text.rsplit(')', 1)[1].split()
    return {'state': fields[0], 'minor_faults': int(fields[7]), 'major_faults': int(fields[9]),
            'utime_ticks': int(fields[11]), 'stime_ticks': int(fields[12]), 'start_time_ticks': int(fields[19]),
            'processor': int(fields[36])}


def snapshot(pid, include_stacks=False, max_threads=128):
    proc = Path('/proc') / str(pid)
    now = time.monotonic_ns()
    stat, err = read_text(proc / 'stat')
    if stat is None:
        return {'monotonic_ns': now, 'pid': pid, 'event': 'process-unavailable', 'error': err}
    io, io_error = read_text(proc / 'io')
    status, status_error = read_text(proc / 'status')
    cgroup, cgroup_error = read_text(proc / 'cgroup')
    result = {'monotonic_ns': now, 'pid': pid, 'event': 'sample',
              'clock_ticks_per_second': os.sysconf('SC_CLK_TCK'),
              'process_stat': stat_fields(stat), 'process_status': status_fields(status),
              'process_io': ({key: int(value) for line in io.splitlines()
                              for key, value in [line.split(':', 1)]} if io is not None else None),
              'cgroup': cgroup, 'errors': {'io': io_error, 'status': status_error, 'cgroup': cgroup_error},
              'threads': []}
    try:
        tids = sorted((proc / 'task').iterdir(), key=lambda p: int(p.name))
    except OSError as exc:
        result['errors']['task'] = type(exc).__name__
        return result
    result['threads_truncated'] = len(tids) > max_threads
    for task in tids[:max_threads]:
        stat, stat_error = read_text(task / 'stat')
        status, status_error = read_text(task / 'status')
        wchan, wchan_error = read_text(task / 'wchan')
        thread = {'tid': int(task.name), 'stat': stat_fields(stat),
                  'status': status_fields(status), 'wchan': wchan.strip() if wchan is not None else None,
                  'errors': {'stat': stat_error, 'status': status_error, 'wchan': wchan_error}}
        if include_stacks:
            stack, stack_error = read_text(task / 'stack')
            thread['kernel_stack'] = stack
            thread['errors']['kernel_stack'] = stack_error
        result['threads'].append(thread)
    result['collection_ns'] = time.monotonic_ns() - now
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('pid', type=int)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--duration', type=float, default=90)
    ap.add_argument('--interval', type=float, default=.2)
    ap.add_argument('--stop-file', type=Path, help='exit gracefully when this path appears')
    ap.add_argument('--stacks', action='store_true', help='best effort; permission failures remain explicit')
    a = ap.parse_args()
    if a.pid <= 0 or not 0 < a.duration <= 120 or not .1 <= a.interval <= 2:
        ap.error('pid>0; duration in (0,120]; interval in [0.1,2] required')
    deadline = time.monotonic() + a.duration
    start_time = None
    with a.output.open('w') as out:
        while time.monotonic() < deadline and not (a.stop_file and a.stop_file.exists()):
            row = snapshot(a.pid, a.stacks)
            if row['event'] == 'sample':
                current_start = row['process_stat']['start_time_ticks']
                if start_time is None:
                    start_time = current_start
                elif current_start != start_time:
                    row = {'monotonic_ns': row['monotonic_ns'], 'pid': a.pid, 'event': 'pid-reused'}
            out.write(json.dumps(row, separators=(',', ':')) + '\n')
            out.flush()
            if row['event'] != 'sample':
                break
            time.sleep(min(a.interval, max(0, deadline - time.monotonic())))


if __name__ == '__main__':
    main()
