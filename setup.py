#!/usr/bin/env python
"""
IOTA FRAMEWORK -- FIRST-RUN SETUP
==================================
Runs automatically on first launch when .iota_env.json is missing.
Can be re-run manually at any time: python setup.py

Handles:
  - Dependency installation (system-wide OR self-contained .venv)
  - GPU / VRAM detection
  - iota.bat + iota.sh launcher generation
  - .iota_env.json creation in project root

.iota_env.json schema:
  {
    "install_mode": "system" | "venv",
    "python_path":  "<path to python executable>",
    "venv_path":    "<path to .venv>",          # venv mode only
    "gpu_name":     "<GPU name or 'cpu'>",
    "vram_gb":      <float>,                    # 0.0 if no GPU
    "setup_timestamp": "<ISO datetime>"
  }

Model-specific data (calibration, queue state) lives at:
  {root}/{family}/{size}/
"""

import os
import sys
import json
import subprocess
import datetime
import platform
import shutil

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ENV_FILE    = os.path.join(SCRIPT_DIR, ".iota_env.json")
VENV_DIR    = os.path.join(SCRIPT_DIR, ".venv")
BAT_FILE    = os.path.join(SCRIPT_DIR, "iota.bat")
SH_FILE     = os.path.join(SCRIPT_DIR, "iota.sh")

W = 64

REQUIRED_PACKAGES = [
    ("torch",           "torch"),
    ("transformers",    "transformers"),
    ("accelerate",      "accelerate"),
    ("bitsandbytes",    "bitsandbytes"),
    ("huggingface_hub", "huggingface_hub"),
    ("numpy",           "numpy"),
    ("pandas",          "pandas"),
    ("scipy",           "scipy"),
    ("scikit-learn",    "sklearn"),
    ("statsmodels",     "statsmodels"),
    ("matplotlib",      "matplotlib"),
    ("seaborn",         "seaborn"),
    ("nvidia-ml-py",    "pynvml"),
    ("flask",           "flask"),
    ("jsonschema",      "jsonschema"),  # v0.83.3: full schema validation in results_schema.validate_results
]

# Packages installed with a custom index URL (CUDA torch build).
# Installed separately before the rest so they're available for GPU detection.
TORCH_PACKAGES = ["torch", "torchvision", "torchaudio"]

# Map nvidia-smi CUDA version string -> PyTorch wheel index URL
TORCH_CUDA_URLS = {
    "12": "https://download.pytorch.org/whl/cu121",
    "11": "https://download.pytorch.org/whl/cu118",
}
TORCH_CPU_URL = "https://download.pytorch.org/whl/cpu"


def detect_cuda_version():
    """Return major CUDA version string ('12', '11', etc.) or None if no GPU."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi"], stderr=subprocess.DEVNULL
        ).decode()
        import re
        m = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', out)
        if m:
            return m.group(1)  # '12', '11', etc.
    except Exception:
        pass
    return None


def torch_has_cuda():
    """Return True if the installed torch was built with CUDA support.
    Never delete and re-import torch in the same process -- doing so re-executes
    torch/__init__.py and re-registers the triton namespace, causing RuntimeError.
    """
    try:
        if 'torch' in sys.modules:
            return sys.modules['torch'].cuda.is_available()
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def install_torch_cuda(cuda_major, pip_exe):
    """Install torch+torchvision+torchaudio with the correct CUDA index URL."""
    url = TORCH_CUDA_URLS.get(cuda_major, TORCH_CUDA_URLS["12"])
    ok(f"CUDA {cuda_major}.x detected -- installing CUDA-enabled torch from {url}")
    try:
        subprocess.check_call(
            pip_exe + ["install"] + TORCH_PACKAGES +
            ["--index-url", url, "--quiet"],
        )
        return True
    except subprocess.CalledProcessError as e:
        err(f"torch CUDA install failed: {e}")
        return False

# ─────────────────────────────────────────────
# CHROME
# ─────────────────────────────────────────────
def bar():
    """Print a thin horizontal rule of width W."""
    print("─" * W)

def dbar():
    """Print a heavy horizontal rule (double line) of width W."""
    print("═" * W)

def blank():
    """Print a blank line."""
    print()

def msg(t):
    """Print a plain indented message."""
    print(f"  {t}")

def ok(t):
    """Print a success-tagged line."""
    print(f"  [+] {t}")

def warn(t):
    """Print a warning-tagged line."""
    print(f"  [!] {t}")

def err(t):
    """Print an error-tagged line."""
    print(f"  [x] {t}")

def header():
    """Print the IOTA setup wizard header banner."""
    dbar()
    msg("  IOTA FRAMEWORK -- SETUP")
    dbar()

def confirm(prompt, default_yes=True):
    """Y/N prompt. Empty input returns default_yes. This is a local
    copy of ui.confirm -- kept here so setup.py can run before ui's
    dependencies are installed."""
    hint = "[Y/n]" if default_yes else "[y/N]"
    raw  = input(f"  {prompt} {hint} > ").strip().lower()
    if not raw: return default_yes
    return raw in ("y", "yes")


# ─────────────────────────────────────────────
# GPU DETECTION
# ─────────────────────────────────────────────
def detect_gpu():
    """Return (gpu_name, vram_gb). Falls back gracefully to ('cpu', 0.0)."""
    try:
        import torch
        if torch.cuda.is_available():
            props    = torch.cuda.get_device_properties(0)
            name     = props.name
            vram_gb  = round(props.total_memory / 1e9, 1)
            return name, vram_gb
    except Exception:
        pass
    # Try nvidia-smi directly (works even without torch)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        ).decode().strip().splitlines()[0]
        name, mem_mb = out.split(",")
        return name.strip(), round(float(mem_mb.strip()) / 1024, 1)
    except Exception:
        pass
    return "cpu", 0.0


# ─────────────────────────────────────────────
# DEPENDENCY CHECK
# ─────────────────────────────────────────────
def check_missing():
    """Return list of REQUIRED_PACKAGES that cannot be imported.
    Skips TORCH_PACKAGES -- torch is handled separately via
    install_torch_cuda because CUDA index URL selection depends on
    detected driver version and the plain pip flow would install CPU-
    only torch by default."""
    import importlib
    missing = []
    for pkg_name, import_name in REQUIRED_PACKAGES:
        if pkg_name in TORCH_PACKAGES:
            continue  # torch handled separately via CUDA-aware installer
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pkg_name)
    return missing


def check_torch_needs_cuda_reinstall():
    """Return True if torch is installed but WITHOUT CUDA support."""
    try:
        import importlib
        importlib.import_module("torch")
    except ImportError:
        return False   # not installed at all -- install_torch_cuda will handle it
    return not torch_has_cuda()


# ─────────────────────────────────────────────
# INSTALL -- SYSTEM WIDE
# ─────────────────────────────────────────────
def install_system(packages, cuda_major):
    """System-wide pip install: torch first with CUDA index URL,
    then the rest of REQUIRED_PACKAGES.

    CPU-only fallback triggers when cuda_major is falsy -- warns the
    user explicitly because IOTA runs are extremely slow without a
    GPU. Each non-torch package tries --break-system-packages first
    (PEP 668) and falls back to plain install on failure."""
    pip_exe = [sys.executable, "-m", "pip"]
    # ── Torch first, with CUDA index URL ──────────────────────────────
    if cuda_major:
        install_torch_cuda(cuda_major, pip_exe)
    else:
        warn("No CUDA detected -- installing CPU-only torch.")
        warn("Runs will be extremely slow without a GPU.")
        try:
            subprocess.check_call(
                pip_exe + ["install"] + TORCH_PACKAGES +
                ["--index-url", TORCH_CPU_URL, "--quiet"],
            )
        except Exception as e:
            err(f"torch CPU install failed: {e}")
    # ── Remaining packages ────────────────────────────────────────────
    for pkg in packages:
        if pkg in TORCH_PACKAGES:
            continue
        msg(f"  Installing {pkg}...")
        try:
            subprocess.check_call(
                pip_exe + ["install", pkg, "--quiet",
                           "--break-system-packages"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            ok(pkg)
        except subprocess.CalledProcessError:
            try:
                subprocess.check_call(
                    pip_exe + ["install", pkg, "--quiet"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                ok(pkg)
            except Exception as e:
                err(f"Failed to install {pkg}: {e}")


# ─────────────────────────────────────────────
# INSTALL -- VENV
# ─────────────────────────────────────────────
def create_venv():
    """Create a .venv/ in the iota root using the current Python."""
    msg(f"Creating virtual environment at {VENV_DIR} ...")
    subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
    ok("Virtual environment created")


def venv_python():
    """Return the platform-correct Python executable path inside
    the .venv/. Windows puts it at Scripts/python.exe; Unix puts it
    at bin/python."""
    if platform.system() == "Windows":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")


def install_venv(packages, cuda_major):
    """Install REQUIRED_PACKAGES into .venv/. Same torch-first-then-
    rest pattern as install_system but using the venv's pip binary
    directly so the outer environment is never touched."""
    py = venv_python()
    pip_exe = [py, "-m", "pip"]
    # Upgrade pip first
    subprocess.call(pip_exe + ["install", "--upgrade", "pip", "--quiet"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # ── Torch first, with CUDA index URL ──────────────────────────────
    if cuda_major:
        install_torch_cuda(cuda_major, pip_exe)
    else:
        warn("No CUDA detected -- installing CPU-only torch.")
        try:
            subprocess.check_call(
                pip_exe + ["install"] + TORCH_PACKAGES +
                ["--index-url", TORCH_CPU_URL, "--quiet"],
            )
        except Exception as e:
            err(f"torch CPU install failed: {e}")
    # ── Remaining packages ────────────────────────────────────────────
    for pkg in packages:
        if pkg in TORCH_PACKAGES:
            continue
        msg(f"  Installing {pkg} into venv...")
        try:
            subprocess.check_call(
                pip_exe + ["install", pkg, "--quiet"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            ok(pkg)
        except Exception as e:
            err(f"Failed: {pkg}: {e}")


# ─────────────────────────────────────────────
# LAUNCHER FILES
# ─────────────────────────────────────────────
def write_bat(mode):
    """Write iota.bat -- the Windows launcher. In 'venv' mode the .bat
    calls .venv\\Scripts\\python.exe directly; in 'system' mode it
    calls the ambient python. 'pause' keeps the window open after
    run.py exits so users can read final output."""
    if mode == "venv":
        content = (
            "@echo off\n"
            "cd /d \"%~dp0\"\n"
            ".venv\\Scripts\\python.exe start_here.py %*\n"
            "pause\n"
        )
    else:
        content = (
            "@echo off\n"
            "cd /d \"%~dp0\"\n"
            "python start_here.py %*\n"
            "pause\n"
        )
    with open(BAT_FILE, "w") as f:
        f.write(content)
    ok(f"iota.bat written")


def write_sh(mode):
    """Write iota.sh -- the Unix launcher. Same mode split as write_bat.
    chmod 755 so it can be run directly without an explicit 'bash'."""
    if mode == "venv":
        content = (
            "#!/bin/bash\n"
            "cd \"$(dirname \"$0\")\"\n"
            ".venv/bin/python start_here.py \"$@\"\n"
        )
    else:
        content = (
            "#!/bin/bash\n"
            "cd \"$(dirname \"$0\")\"\n"
            "python3 start_here.py \"$@\"\n"
        )
    with open(SH_FILE, "w") as f:
        f.write(content)
    try:
        os.chmod(SH_FILE, 0o755)
    except Exception:
        pass
    ok(f"iota.sh written")


# ─────────────────────────────────────────────
# SELF-RELAUNCH CHECK
# ─────────────────────────────────────────────



# ─────────────────────────────────────────────
# WRITE ENV FILE
# ─────────────────────────────────────────────
def write_env(mode, python_path, gpu_name, vram_gb):
    """Write .iota_env.json -- the one-time install manifest.

    Consumed by cartography.load_env() / get_vram_gb() and by the
    dashboard's VRAM checks (ui.estimate_model_vram_gb falls back to
    this file when torch.cuda isn't available, e.g. in the Flask
    subprocess before a model is loaded). Written exactly once by
    run_setup() on first launch and updated on any --force re-run."""
    data = {
        "install_mode":      mode,
        "python_path":       python_path,
        "venv_path":         VENV_DIR if mode == "venv" else "",
        "gpu_name":          gpu_name,
        "vram_gb":           vram_gb,
        "setup_timestamp":   datetime.datetime.now().isoformat(),
    }
    with open(ENV_FILE, "w") as f:
        json.dump(data, f, indent=2)
    ok(f".iota_env.json written")
    return data


# ─────────────────────────────────────────────
# LOAD ENV
# ─────────────────────────────────────────────
def load_env():
    """Return env dict or None if not set up."""
    if not os.path.exists(ENV_FILE):
        return None
    try:
        return json.load(open(ENV_FILE))
    except Exception:
        return None


# ─────────────────────────────────────────────
# MAIN SETUP FLOW
# ─────────────────────────────────────────────
def run_setup(force=False):
    """
    Run the first-time setup wizard.
    force=True to re-run even if .iota_env.json already exists.
    Returns the env dict on success.
    """
    if not force and os.path.exists(ENV_FILE):
        # Even on cached setup, silently fix CPU-only torch if CUDA is available
        cuda_major = detect_cuda_version()
        if cuda_major and check_torch_needs_cuda_reinstall():
            blank()
            warn("torch is installed but WITHOUT CUDA support (CPU-only build).")
            warn(f"CUDA {cuda_major}.x detected -- reinstalling CUDA-enabled torch...")
            blank()
            pip_exe = [sys.executable, "-m", "pip"]
            if install_torch_cuda(cuda_major, pip_exe):
                ok("CUDA torch installed successfully.")
                ok("Restarting to load the new torch build...")
                blank()
                import subprocess as _sp
                _sp.Popen([sys.executable] + sys.argv)
                sys.exit(0)
            else:
                err("Reinstall failed. Run manually:")
                err(f"  pip install torch torchvision torchaudio "
                    f"--index-url {TORCH_CUDA_URLS.get(cuda_major, TORCH_CUDA_URLS['12'])}")
            blank()
        return load_env()

    header()
    blank()
    msg("Welcome to IOTA Framework.")
    msg("This setup runs once. It installs dependencies and detects your GPU.")
    blank()

    # ── Detect CUDA early -- needed to pick torch build ─────────────────
    cuda_major = detect_cuda_version()
    if cuda_major:
        ok(f"CUDA {cuda_major}.x detected via nvidia-smi")
    else:
        warn("nvidia-smi not found or no GPU -- will install CPU-only torch")
    blank()

    # ── Install deps? ──────────────────────────────────────────────────
    missing = check_missing()
    torch_needs_cuda_fix = check_torch_needs_cuda_reinstall()
    needs_install = bool(missing) or torch_needs_cuda_fix

    if needs_install:
        if missing:
            msg(f"Missing packages ({len(missing)}): {', '.join(missing)}")
        if torch_needs_cuda_fix:
            warn("torch installed but CPU-only -- will reinstall with CUDA support")
        blank()
        if not confirm("Install / fix dependencies?", default_yes=True):
            warn("Cannot proceed without dependencies.")
            warn("Run setup again when ready.")
            sys.exit(1)
        blank()

        # ── System or venv? ───────────────────────────────────────────
        bar()
        msg("Install mode:")
        blank()
        msg("  [1]  System-wide   -- installs into your current Python environment")
        msg("           Simpler. Works if you only use one Python.")
        blank()
        msg("  [2]  Self-contained -- creates a .venv folder in this directory")
        msg("           Nothing touches your system. Recommended if unsure.")
        blank()
        while True:
            raw = input("  > ").strip()
            if raw in ("1", "2"):
                break
            warn("Enter 1 or 2.")

        mode = "system" if raw == "1" else "venv"
        blank()

        if mode == "venv":
            if not os.path.exists(VENV_DIR):
                create_venv()
            install_venv(missing, cuda_major)
            python_path = venv_python()
        else:
            install_system(missing, cuda_major)
            python_path = sys.executable

        # ── Post-install CUDA verification ────────────────────────────
        blank()
        bar()
        if cuda_major:
            if torch_has_cuda():
                ok("torch.cuda.is_available() = True  \u2713")
            else:
                err("torch.cuda.is_available() = False after install.")
                err("torch may need a matching CUDA runtime. Try:")
                err(f"  pip install torch torchvision torchaudio "
                    f"--index-url {TORCH_CUDA_URLS.get(cuda_major, TORCH_CUDA_URLS['12'])}")

    else:
        blank()
        ok("All dependencies already installed.")
        if cuda_major and torch_has_cuda():
            ok("torch CUDA support confirmed.")
        elif cuda_major:
            warn("torch installed but CUDA unavailable -- check torch build.")
        if ".venv" in sys.executable or "venv" in sys.executable.lower():
            mode        = "venv"
            python_path = sys.executable
        else:
            mode        = "system"
            python_path = sys.executable

    # ── GPU detection ──────────────────────────────────────────────────
    blank()
    bar()
    msg("Detecting GPU...")
    gpu_name, vram_gb = detect_gpu()
    if vram_gb > 0:
        ok(f"GPU: {gpu_name}  ({vram_gb} GB VRAM)")
    else:
        warn("No GPU detected. Experiments will run on CPU (very slow).")

    # ── Write launchers ────────────────────────────────────────────────
    blank()
    bar()
    write_bat(mode)
    write_sh(mode)

    # ── Write env ──────────────────────────────────────────────────────
    env = write_env(mode, python_path, gpu_name, vram_gb)

    # ── Summary ────────────────────────────────────────────────────────
    blank()
    dbar()
    ok("Setup complete.")
    blank()
    if platform.system() == "Windows":
        msg("  To start IOTA:  double-click iota.bat")
        msg("                  or run:  iota.bat")
    else:
        msg("  To start IOTA:  ./iota.sh")
        msg("                  or run:  python run.py")
    if vram_gb > 0:
        msg(f"\n  GPU detected: {gpu_name} ({vram_gb} GB)")
        if vram_gb < 8:
            blank()
            warn("Less than 8 GB VRAM detected.")
            warn("You may need to use smaller models (3B or less) or 4-bit quantization.")
    dbar()
    blank()

    return env


# ─────────────────────────────────────────────
# CLI ENTRY
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="IOTA Setup")
    p.add_argument("--force", action="store_true", help="Re-run even if already set up")
    args = p.parse_args()
    run_setup(force=args.force)
