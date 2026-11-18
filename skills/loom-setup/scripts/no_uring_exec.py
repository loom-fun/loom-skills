#!/usr/bin/env python3
"""Exec one owned SDK child with io_uring_setup denied as ENOSYS.
Equivalent to the documented systemd syscall-filter workaround; not an SDK patch.
No preload, payload copy, host policy change or privilege increase. Fail closed.
"""
import ctypes
import ctypes.util
import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def setup_probe():
    name = ctypes.util.find_library('seccomp')
    if not name:
        raise RuntimeError('libseccomp unavailable')
    seccomp = ctypes.CDLL(name)
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    number = seccomp.seccomp_syscall_resolve_name(b'io_uring_setup')
    if number < 0:
        raise RuntimeError('native io_uring_setup syscall not resolved')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    params = ctypes.create_string_buffer(256)
    ctypes.set_errno(0)
    result = libc.syscall(ctypes.c_long(number), ctypes.c_uint(1), ctypes.byref(params))
    error = ctypes.get_errno()
    if result >= 0:
        os.close(result)
    return {'syscall': number, 'result': result, 'errno': error, 'available': result >= 0}


def disable_io_uring():
    name = ctypes.util.find_library('seccomp')
    if not name:
        raise RuntimeError('libseccomp unavailable; filter was not applied')
    lib = ctypes.CDLL(name, use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_rule_add.restype = ctypes.c_int
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_load.restype = ctypes.c_int
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    number = lib.seccomp_syscall_resolve_name(b'io_uring_setup')
    if number < 0:
        raise RuntimeError('native io_uring_setup syscall not resolved')
    ctx = lib.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    if not ctx:
        raise RuntimeError('seccomp_init failed')
    try:
        result = lib.seccomp_rule_add(ctx, 0x00050000 | errno.ENOSYS, number, 0)
        if result:
            raise RuntimeError(f'seccomp_rule_add failed: {result}')
        result = lib.seccomp_load(ctx)
        if result:
            raise RuntimeError(f'seccomp_load failed: {result}')
    finally:
        lib.seccomp_release(ctx)
    probe = setup_probe()  # Same restricted process that will exec the SDK.
    if (probe['result'], probe['errno']) != (-1, errno.ENOSYS):
        raise RuntimeError(f'filter verification failed: {probe}')
    return {'library': name, 'setup_probe': probe, 'pid': os.getpid(),
            'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def self_test(output, require_distinct=False):
    baseline = setup_probe()
    env = dict(os.environ, LOOM_IO_TEST_SENTINEL='owned-child-only', LOOM_IO_TEST_SYSCALL=str(baseline['syscall']))
    expected = hashlib.sha256(json.dumps(sorted(env.items())).encode()).hexdigest()
    env['LOOM_IO_TEST_EXPECTED_ENV'] = expected
    probe = '''import ctypes,hashlib,json,os,sys
expected=os.environ['LOOM_IO_TEST_EXPECTED_ENV'];env=dict(os.environ);env.pop('LOOM_IO_TEST_EXPECTED_ENV')
libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long;params=ctypes.create_string_buffer(256);ctypes.set_errno(0);uring_result=libc.syscall(ctypes.c_long(int(os.environ['LOOM_IO_TEST_SYSCALL'])),ctypes.c_uint(1),ctypes.byref(params));uring_errno=ctypes.get_errno()
status=open('/proc/self/status').read();fields=dict(line.split(':',1) for line in status.splitlines() if ':' in line)
print(json.dumps({'pid':os.getpid(),'uring_result':uring_result,'uring_errno':uring_errno,'arguments':sys.argv[1:],'environment_equal':hashlib.sha256(json.dumps(sorted(env.items())).encode()).hexdigest()==expected,'read_ok':status.startswith('Name:'),'getpid_ok':os.getpid()>0,'no_new_privs':fields.get('NoNewPrivs','').strip(),'uid':os.getuid(),'gid':os.getgid()}))
'''
    args = ['argument with spaces', 'literal;$VALUE`not-shell`']
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--', sys.executable, '-c', probe, *args],
                             env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = child.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        child.kill()
        stdout, stderr = child.communicate(timeout=2)
        Path(output).write_text(json.dumps({'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                          'baseline_setup': baseline, 'restricted_child_pid': child.pid,
                                          'exec_and_filter_checks_pass': False, 'error': 'restricted child timed out',
                                          'child_stderr': stderr}, indent=1) + '\n')
        raise RuntimeError('restricted exec preflight timed out; child killed; see output')
    result = json.loads(stdout) if child.returncode == 0 else None
    parent_after = setup_probe()
    valid = bool(result and (result['uring_result'], result['uring_errno']) == (-1, errno.ENOSYS)
                 and parent_after['available'] == baseline['available']
                 and (baseline['available'] or parent_after['errno'] == baseline['errno'])
                 and result['pid'] == child.pid and result['arguments'] == args
                 and result['environment_equal'] and result['read_ok'] and result['getpid_ok']
                 and result['no_new_privs'] == '1' and result['uid'] == os.getuid() and result['gid'] == os.getgid())
    report = {'libseccomp': ctypes.util.find_library('seccomp'), 'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'baseline_setup': baseline, 'parent_setup_after_child': parent_after, 'restricted_child_pid': child.pid, 'restricted_child': result,
              'child_returncode': child.returncode, 'child_stderr': stderr,
              'exec_and_filter_checks_pass': valid, 'baseline_distinct': baseline['available']}
    Path(output).write_text(json.dumps(report, indent=1) + '\n')
    if not valid:
        raise RuntimeError('restricted exec preflight failed; see output')
    if require_distinct and not baseline['available']:
        raise RuntimeError('baseline io_uring unavailable: backend comparison is not distinct; no weights should be fetched')


def main():
    args = sys.argv[1:]
    try:
        if args and args[0] == '--self-test':
            if len(args) not in (2, 3) or (len(args) == 3 and args[2] != '--require-distinct-baseline'):
                raise RuntimeError('usage: --self-test OUTPUT [--require-distinct-baseline]')
            self_test(args[1], len(args) == 3)
            return
        if args and args[0] == '--':
            args.pop(0)
        if not args:
            raise RuntimeError('usage: no_uring_exec.py -- COMMAND [ARG ...]')
        record = disable_io_uring()
        record['sdk_command'] = args
        record['scope'] = 'owned SDK child; shared proactor fallback, not file-only'
        output = os.environ.get('LOOM_IO_BACKEND_PROVENANCE')
        if output:
            Path(output).write_text(json.dumps(record, indent=1) + '\n')
        print(f"no_uring_exec: io_uring_setup({record['setup_probe']['syscall']}) verified ENOSYS; exec pid {os.getpid()}", file=sys.stderr, flush=True)
        os.execvp(args[0], args)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise SystemExit(f'no_uring_exec: {exc}')


if __name__ == '__main__':
    main()
