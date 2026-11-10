"""Decode a native Windows server command line without losing quoted paths."""
import ctypes
import sys


def split_command_line(command):
    shell = ctypes.WinDLL("shell32", use_last_error=True)
    shell.CommandLineToArgvW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    shell.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
    count = ctypes.c_int()
    argv = shell.CommandLineToArgvW(command, ctypes.byref(count))
    if not argv:
        raise ctypes.WinError(ctypes.get_last_error())
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    try:
        return [argv[i] for i in range(count.value)]
    finally:
        kernel.LocalFree(argv)


if __name__ == "__main__":
    for arg in split_command_line(sys.stdin.read().strip()):
        sys.stdout.buffer.write(arg.encode("utf-8") + b"\0")
