#!/usr/bin/env python3
"""Read-only GGUF FD/backing metadata. Never reads tensor payloads or copies files."""
import argparse
import json
import os
from pathlib import Path
import stat
import time


def read(path):
    try:
        return path.read_text(), None
    except OSError as exc:
        return None, type(exc).__name__


def mounts(text):
    records = {}
    for line in (text or '').splitlines():
        left, separator, right = line.partition(' - ')
        if not separator:
            continue
        a, b = left.split(), right.split()
        if len(a) >= 6 and len(b) >= 3:
            records[a[0]] = {'device': a[2], 'root': a[3], 'mountpoint': a[4],
                              'mount_options': a[5], 'filesystem': b[0],
                              'source': b[1], 'super_options': b[2]}
    return records


def host_io():
    text, error = read(Path('/proc/meminfo'))
    keys = {'MemTotal', 'MemAvailable', 'Dirty', 'Writeback', 'Shmem', 'SwapTotal', 'SwapFree'}
    memory = {key: value.strip() for line in (text or '').splitlines() if ':' in line
              for key, value in [line.split(':', 1)] if key in keys}
    io, io_error = read(Path('/proc/pressure/io'))
    mem, mem_error = read(Path('/proc/pressure/memory'))
    disabled, disabled_error = read(Path('/proc/sys/kernel/io_uring_disabled'))
    return {'kernel': {'sysname': os.uname().sysname, 'release': os.uname().release, 'machine': os.uname().machine},
            'io_uring_disabled_raw': disabled.strip() if disabled is not None else None,
            'meminfo_raw': memory if text is not None else None,
            'io_pressure_raw': io, 'memory_pressure_raw': mem,
            'errors': {'meminfo': error, 'io_pressure': io_error, 'memory_pressure': mem_error, 'io_uring_disabled': disabled_error}}


def snapshot(pid):
    proc = Path('/proc') / str(pid)
    mount_text, mount_error = read(proc / 'mountinfo')
    known_mounts = mounts(mount_text)
    status, status_error = read(proc / 'status')
    memory_keys = {'VmRSS', 'VmHWM', 'VmData', 'VmSize', 'VmPin', 'VmLck', 'RssAnon', 'RssFile', 'RssShmem', 'VmSwap', 'Cpus_allowed_list'}
    process_memory = {key: value.strip() for line in (status or '').splitlines() if ':' in line
                      for key, value in [line.split(':', 1)] if key in memory_keys}
    limits, limits_error = read(proc / 'limits')
    locked_limit = next((line for line in (limits or '').splitlines() if line.startswith('Max locked memory')), None)
    result = {'pid': pid, 'monotonic_ns': time.monotonic_ns(), 'host_io': host_io(),
              'files': [], 'io_uring_fds': [], 'process_memory_raw': process_memory if status is not None else None,
              'locked_memory_limit_raw': locked_limit,
              'process_security_raw': {key: value.strip() for line in (status or '').splitlines() if ':' in line
                                       for key, value in [line.split(':', 1)] if key in {'Uid', 'Gid', 'NoNewPrivs', 'Seccomp', 'Seccomp_filters'}},
              'errors': {'mountinfo': mount_error, 'status': status_error, 'limits': limits_error}}
    try:
        fds = list((proc / 'fd').iterdir())
    except OSError as exc:
        result['errors']['fd'] = type(exc).__name__
        return result
    for fd in sorted(fds, key=lambda p: int(p.name)):
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if 'io_uring' in target:
            result['io_uring_fds'].append({'fd': int(fd.name), 'target': target})
        if '.gguf' not in target:
            continue
        info, error = read(proc / 'fdinfo' / fd.name)
        fields = dict(line.split(':', 1) for line in (info or '').splitlines() if ':' in line)
        flags = int(fields['flags'], 8) if 'flags' in fields else None
        record = {'fd': int(fd.name), 'target': target, 'fdinfo_error': error,
                  'flags_octal': fields.get('flags', '').strip() or None,
                  'direct': bool(flags & getattr(os, 'O_DIRECT', 0)) if flags is not None else None,
                  'mount': known_mounts.get(fields.get('mnt_id', '').strip())}
        try:
            s = fd.stat()
            record.update(inode=s.st_ino, size_bytes=s.st_size,
                          device=f'{os.major(s.st_dev)}:{os.minor(s.st_dev)}', regular_file=stat.S_ISREG(s.st_mode))
        except OSError as exc:
            record['stat_error'] = type(exc).__name__
        result['files'].append(record)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('pid', type=int)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--require-files', action='store_true')
    a = ap.parse_args()
    result = snapshot(a.pid)
    a.output.write_text(json.dumps(result, indent=1) + '\n')
    if a.require_files and not result['files']:
        raise SystemExit('no GGUF file-descriptor metadata captured; see output errors')


if __name__ == '__main__':
    main()
