#!/usr/bin/env python3
"""What measured a result: the device and the Loom SDK (hrx-loom) that served the model. Standard library only.

usage: machine_info.py [device|hrx-loom|all]      prints JSON (default: all)

  device    {cpu, gpu, os} of this machine (detect_device). Every value comes from the hardware or the OS
            release, never from kernel, driver, ROCm or patch versions, so the same machine and OS release
            always give the same strings.
  hrx-loom  the installed Loom SDK (hrx_loom_identity): $LOOM_ROOT/loom-install.json as install_loom.sh wrote
            it, else the installed version plus a fingerprint of its dist-info RECORD (marked unverified).

The card scripts (card/make_info.py perf/gates, loom_factory.py, the nightly gates) call these on the machine
that measured and store the result with each evidence source. detect_device() is ported from danielholanda's
loom_factory.py draft (loom-skills PR #35, "loom-info/1").
"""
import glob
import hashlib
import json
import os
import platform
import re
import sys

AMD_PCI_VENDOR = "1002"
# libdrm's PCI id -> marketing name table; the distribution's copy first, then AMD's driver packages.
AMDGPU_IDS = ["/usr/share/libdrm/amdgpu.ids", "/opt/amdgpu/share/libdrm/amdgpu.ids"]


def _clean(name):
    """Hardware names without trademark marks or padding: 'AMD Radeon(TM) 8060S  ' -> 'AMD Radeon 8060S'."""
    return " ".join(re.sub(r"\((?:TM|R|C)\)", "", name, flags=re.I).split())


def _linux_gpus():
    names = set()
    for dev in glob.glob("/sys/bus/pci/devices/*"):
        def attr(k):
            try:
                with open(os.path.join(dev, k)) as f:
                    v = f.read().strip().lower()
                    return v[2:] if v.startswith("0x") else v
            except OSError:
                return ""
        if attr("vendor") != AMD_PCI_VENDOR or not attr("class").startswith("03"):  # 03xxxx: display controller
            continue
        did, rev, name = attr("device"), attr("revision"), None
        for path in AMDGPU_IDS:
            try:
                with open(path, encoding="utf-8") as f:
                    rows = [line.split(",", 2) for line in f if not line.startswith("#") and line.count(",") >= 2]
            except OSError:
                continue
            rows = [(d.strip().lower(), r.strip().lower(), n.strip()) for d, r, n in rows]
            # One device id covers several products (1586 is the 8060S, 8050S or 8040S by revision).
            name = next((n for d, r, n in rows if d == did and r == rev), None)
            if name:
                break
        names.add(name or f"AMD GPU [{AMD_PCI_VENDOR}:{did}:{rev}]")
    return names


def _windows_gpus():
    import winreg
    names = set()
    cls = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"  # display adapters
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, cls) as root:
        for i in range(winreg.QueryInfoKey(root)[0]):
            try:
                with winreg.OpenKey(root, winreg.EnumKey(root, i)) as k:
                    if f"ven_{AMD_PCI_VENDOR}" in winreg.QueryValueEx(k, "MatchingDeviceId")[0].lower():
                        names.add(winreg.QueryValueEx(k, "DriverDesc")[0])
            except OSError:
                continue
    return names


def detect_device():
    """{cpu, gpu, os} of this machine:
      cpu  the CPUID brand string (/proc/cpuinfo or the Windows registry report the same one), trademarks stripped
      gpu  the marketing name of each AMD display adapter, by PCI id and revision (libdrm amdgpu.ids on Linux,
           the driver's DriverDesc on Windows), as 'AMD ...', sorted and joined with ' + '; omitted if none
      os   Linux: os-release NAME and VERSION_ID ('Ubuntu 24.04', not the 24.04.3 point release);
           Windows: 'Windows 10' or 'Windows 11' by build, plus the feature update ('Windows 11 25H2')"""
    cpu, gpus, os_name = "", set(), f"{platform.system()} {platform.release()}"
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                cpu = next((line.split(":", 1)[1] for line in f if line.startswith("model name")), "")
        except OSError:
            pass
        rel = {}
        for path in ("/etc/os-release", "/usr/lib/os-release"):
            try:
                with open(path, encoding="utf-8") as f:
                    rel = dict(line.rstrip("\n").split("=", 1) for line in f if "=" in line)
                break
            except OSError:
                continue
        rel = {k: v.strip('"') for k, v in rel.items()}
        if rel.get("NAME"):
            os_name = " ".join(filter(None, (rel["NAME"], rel.get("VERSION_ID"))))
        gpus = _linux_gpus()
    elif sys.platform == "win32":
        import winreg
        hk = winreg.HKEY_LOCAL_MACHINE
        with winreg.OpenKey(hk, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0]
        # ProductName says "Windows 10" on Windows 11 too; the build number tells them apart.
        with winreg.OpenKey(hk, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as k:
            build = int(winreg.QueryValueEx(k, "CurrentBuildNumber")[0])
            try:
                feature = winreg.QueryValueEx(k, "DisplayVersion")[0]
            except OSError:
                feature = str(build)
        os_name = f"Windows {11 if build >= 22000 else 10} {feature}"
        gpus = _windows_gpus()
    # libdrm says "Radeon 8060S Graphics" where the Windows driver says "AMD Radeon(TM) 8060S Graphics".
    gpus = {g if g.startswith("AMD ") else f"AMD {g}" for g in map(_clean, gpus)}
    device = {"cpu": _clean(cpu or platform.processor()), "gpu": " + ".join(sorted(gpus)), "os": os_name}
    return {k: v for k, v in device.items() if v}


def _dist_info(venv):
    pats = [os.path.join(venv, "lib", "python*", "site-packages", "hrx_loom-*.dist-info"),
            os.path.join(venv, "Lib", "site-packages", "hrx_loom-*.dist-info")]
    found = sorted(p for pat in pats for p in glob.glob(pat))
    return found[0] if found else None


def hrx_loom_identity(loom_root=None, venv=None):
    """The installed Loom SDK. install_loom.sh's $LOOM_ROOT/loom-install.json when present: {hrx_loom (version),
    source (pypi | ci), hrx_system_sha, artifact_id, branch, wheel_sha256}. Otherwise the installed version and
    the first 16 hex of sha256(dist-info RECORD), "verified": false (a RECORD names every installed file with
    its hash, so the same wheel installed the same way gives the same fingerprint). None when nothing is found."""
    loom_root = loom_root or os.environ.get("LOOM_ROOT")
    venv = venv or os.environ.get("VENV") or (os.path.join(loom_root, ".venv") if loom_root else None)
    if loom_root:
        p = os.path.join(loom_root, "loom-install.json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    d = _dist_info(venv) if venv else None
    if not d:
        return None
    version = os.path.basename(d)[len("hrx_loom-"):-len(".dist-info")]
    with open(os.path.join(d, "RECORD"), "rb") as f:
        fp = hashlib.sha256(f.read()).hexdigest()[:16]
    return {"hrx_loom": version, "fingerprint": fp, "verified": False}


def main(argv=None):
    what = (argv or sys.argv[1:] or ["all"])[0]
    out = {"device": detect_device, "hrx-loom": hrx_loom_identity,
           "all": lambda: {"device": detect_device(), "hrx_loom": hrx_loom_identity()}}[what]()
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
