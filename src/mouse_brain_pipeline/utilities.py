"""Cross-cutting helpers: environment checks, logging, path safety.

Deliberately importable with only the standard library. ``psutil`` is used when
present for richer RAM/CPU info but is optional.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LOG = logging.getLogger("mouse_brain_pipeline")


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def human_bytes(n: float | None) -> str:
    if n is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    f = float(n)
    for u in units:
        if abs(f) < 1024.0 or u == units[-1]:
            return f"{f:.1f} {u}"
        f /= 1024.0
    return f"{f:.1f} PB"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def has_space_in_path(path: str | Path | None) -> bool:
    """True if the path contains a space (problematic for Brainmapper/Cellfinder)."""
    return path is not None and " " in str(path)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(log_file: str | Path | None = None, verbose: bool = False) -> logging.Logger:
    """Configure the package logger to stream to stderr and (optionally) a file."""
    level = logging.DEBUG if verbose else logging.INFO
    LOG.setLevel(level)
    LOG.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    LOG.addHandler(stream)

    if log_file is not None:
        ensure_dir(Path(log_file).parent)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        LOG.addHandler(fh)
    LOG.propagate = False
    return LOG


# --------------------------------------------------------------------------- #
# Environment inspection
# --------------------------------------------------------------------------- #
@dataclass
class EnvironmentReport:
    python_version: str
    python_executable: str
    platform: str
    os_release: str
    cpu_logical: int | None
    ram_total: int | None
    ram_available: int | None
    cuda_available: bool
    cuda_info: str
    disk: dict[str, dict[str, int | None]]
    packages: dict[str, str | None]
    brainmapper_installed: bool
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ram_info() -> tuple[int | None, int | None]:
    try:
        import psutil  # noqa: PLC0415

        vm = psutil.virtual_memory()
        return vm.total, vm.available
    except Exception:
        pass
    # Linux/macOS: total via sysconf.
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return total, None
    except (ValueError, AttributeError, OSError):
        pass
    # Windows fallback via ctypes (no psutil needed).
    if os.name == "nt":
        try:
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullTotalPhys), int(stat.ullAvailPhys)
        except Exception:
            pass
    return None, None


def get_cuda_info() -> tuple[bool, str]:
    """Detect an NVIDIA GPU via ``nvidia-smi`` (no torch/cupy dependency)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False, "nvidia-smi not found (no NVIDIA driver detected)"
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode == 0 and out.stdout.strip():
            return True, out.stdout.strip().replace("\n", " | ")
        return False, f"nvidia-smi returned code {out.returncode}"
    except Exception as exc:  # pragma: no cover
        return False, f"nvidia-smi error: {exc}"


def check_packages() -> dict[str, str | None]:
    """Return {import_name: version-or-None} for the key dependencies."""
    import importlib

    names = [
        "numpy",
        "tifffile",
        "imagecodecs",
        "dask",
        "zarr",
        "scipy",
        "skimage",
        "yaml",
        "psutil",
        "PIL",
        "matplotlib",
        "pandas",
        "tqdm",
        "napari",
        "cellfinder",
        "brainglobe_workflows",
        "brainglobe_atlasapi",
        "brainreg",
    ]
    out: dict[str, str | None] = {}
    for name in names:
        try:
            mod = importlib.import_module(name)
            out[name] = getattr(mod, "__version__", "installed")
        except Exception:
            out[name] = None
    return out


def check_brainmapper_installed() -> bool:
    """True if the ``brainmapper`` CLI is on PATH."""
    return shutil.which("brainmapper") is not None


def check_environment(extra_paths: list[str | Path] | None = None) -> EnvironmentReport:
    """Gather RAM/CPU/disk/OS/Python/CUDA and key package availability."""
    notes: list[str] = []
    ram_total, ram_avail = _ram_info()
    cuda_avail, cuda_info = get_cuda_info()
    packages = check_packages()

    # Disk usage for relevant mounts plus any caller-supplied paths.
    disk: dict[str, dict[str, int | None]] = {}
    candidate_paths: list[Path] = [Path.cwd()]
    if extra_paths:
        candidate_paths += [Path(p) for p in extra_paths if p]
    seen_roots: set[str] = set()
    for p in candidate_paths:
        try:
            anchor = str(Path(p).expanduser().anchor or Path(p).expanduser())
        except Exception:
            continue
        if anchor in seen_roots:
            continue
        seen_roots.add(anchor)
        try:
            usage = shutil.disk_usage(p if Path(p).exists() else Path(p).anchor or ".")
            disk[anchor] = {"total": usage.total, "used": usage.used, "free": usage.free}
        except Exception:
            disk[anchor] = {"total": None, "used": None, "free": None}

    py = sys.version_info
    if py < (3, 10) or py >= (3, 12):
        notes.append(
            f"Python {py.major}.{py.minor} detected. The BrainGlobe/Cellfinder stack "
            "targets Python 3.10/3.11. Create a clean 3.11 env for full analysis."
        )
    if packages.get("numpy") is None:
        notes.append("Core scientific packages are not installed yet (numpy missing).")
    if not cuda_avail:
        notes.append("No CUDA GPU detected -- Cellfinder will run on CPU (slow).")

    return EnvironmentReport(
        python_version=f"{py.major}.{py.minor}.{py.micro}",
        python_executable=sys.executable,
        platform=platform.platform(),
        os_release=f"{platform.system()} {platform.release()}",
        cpu_logical=os.cpu_count(),
        ram_total=ram_total,
        ram_available=ram_avail,
        cuda_available=cuda_avail,
        cuda_info=cuda_info,
        disk=disk,
        packages=packages,
        brainmapper_installed=check_brainmapper_installed(),
        notes=notes,
    )


def print_environment_report(report: EnvironmentReport | None = None) -> EnvironmentReport:
    """Pretty-print the environment report to stdout and return it."""
    report = report or check_environment()
    print("=" * 70)
    print("ENVIRONMENT REPORT")
    print("=" * 70)
    print(f"Python        : {report.python_version}  ({report.python_executable})")
    print(f"Platform      : {report.platform}")
    print(f"CPU (logical) : {report.cpu_logical}")
    print(f"RAM total     : {human_bytes(report.ram_total)}")
    print(f"RAM available : {human_bytes(report.ram_available)}")
    print(f"CUDA GPU      : {'YES' if report.cuda_available else 'no'} -- {report.cuda_info}")
    print(f"brainmapper   : {'on PATH' if report.brainmapper_installed else 'NOT installed'}")
    print("-" * 70)
    print("Disk:")
    for root, u in report.disk.items():
        print(f"  {root:<8} free {human_bytes(u['free'])} / total {human_bytes(u['total'])}")
    print("-" * 70)
    print("Key packages:")
    for name, ver in report.packages.items():
        mark = ver if ver else "MISSING"
        print(f"  {name:<22} {mark}")
    if report.notes:
        print("-" * 70)
        print("Notes:")
        for n in report.notes:
            print(f"  * {n}")
    print("=" * 70)
    return report


# --------------------------------------------------------------------------- #
# RAM safety estimate for a single plane / stack
# --------------------------------------------------------------------------- #
def estimate_plane_bytes(shape: tuple[int, int] = (13912, 9906), dtype_bytes: int = 2) -> int:
    """Bytes for one full-resolution plane (default ~ the confirmed TIFF size)."""
    h, w = shape
    return h * w * dtype_bytes


def warn_if_low_memory(planes_in_ram: int, shape: tuple[int, int] = (13912, 9906)) -> list[str]:
    """Return warnings if holding ``planes_in_ram`` planes risks exhausting RAM."""
    warnings: list[str] = []
    _, avail = _ram_info()
    need = estimate_plane_bytes(shape) * planes_in_ram
    if avail is not None and need > 0.5 * avail:
        warnings.append(
            f"Holding {planes_in_ram} full planes needs ~{human_bytes(need)}; "
            f"available RAM ~{human_bytes(avail)}. Reduce tile size / plane count."
        )
    return warnings
