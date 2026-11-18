"""Send a console control event to one process (Windows): attach to its console, ignore the event here, raise it.
usage: python ctrlc.py PID EVENT   (EVENT 0 = CTRL_C, 1 = CTRL_BREAK)"""
import ctypes
import sys
import time

k = ctypes.windll.kernel32
k.FreeConsole()
if not k.AttachConsole(int(sys.argv[1])):
    print("AttachConsole failed", ctypes.GetLastError())
    sys.exit(2)
k.SetConsoleCtrlHandler(None, True)
ok = k.GenerateConsoleCtrlEvent(int(sys.argv[2]), 0)
time.sleep(2)
k.FreeConsole()
sys.exit(0 if ok else 3)
