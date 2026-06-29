import os
import platform


THREAD_KEYS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]


def _cpu_brand():
    name = platform.processor()
    if name:
        return name
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        return None
    return None


def _blas_info():
    try:
        import numpy
        cfg = numpy.show_config(mode="dicts")
    except Exception:
        return None
    out = {}
    build = cfg.get("Build Dependencies", {})
    for key in ("blas", "lapack"):
        dep = build.get(key, {})
        out[key] = {
            "name": dep.get("name"),
            "version": dep.get("version"),
        }
    simd = cfg.get("SIMD Extensions", {})
    out["simd_baseline"] = simd.get("baseline")
    out["simd_found"] = simd.get("found")
    return out


def _packages():
    try:
        from importlib import metadata
        dists = list(metadata.distributions())
    except Exception:
        return None
    found = {}
    for dist in dists:
        try:
            name = dist.metadata["Name"]
            ver = dist.version
        except Exception:
            continue
        if name is not None:
            found[name] = ver
    return dict(sorted(found.items(), key=lambda kv: kv[0].lower()))


def collect():
    info = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "os_system": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "architecture": platform.architecture()[0],
        "cpu": _cpu_brand(),
        "thread_env": {key: os.environ.get(key) for key in THREAD_KEYS},
    }
    try:
        import numpy
        info["numpy_version"] = numpy.__version__
    except Exception:
        info["numpy_version"] = None
    try:
        from importlib import metadata as _mpl_md
        info["matplotlib_version"] = _mpl_md.version("matplotlib")
    except Exception:
        try:
            import matplotlib
            info["matplotlib_version"] = matplotlib.__version__
        except Exception:
            info["matplotlib_version"] = None
    info["blas"] = _blas_info()
    info["packages"] = _packages()
    return info


def summary_lines(info):
    blas = info.get("blas") or {}
    blas_blas = blas.get("blas") or {}
    return [
        "os           = " + str(info.get("platform")),
        "machine      = " + str(info.get("machine")) + " (" + str(info.get("architecture")) + ")",
        "cpu          = " + str(info.get("cpu")),
        "python       = " + str(info.get("python_version")) + " " + str(info.get("python_implementation")),
        "numpy        = " + str(info.get("numpy_version")),
        "matplotlib   = " + str(info.get("matplotlib_version")),
        "blas         = " + str(blas_blas.get("name")) + " " + str(blas_blas.get("version")),
        "thread_env   = " + str(info.get("thread_env")),
    ]
