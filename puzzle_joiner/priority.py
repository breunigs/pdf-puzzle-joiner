import os
import shutil


def _build_low_prio_prefix():
    """Build command prefix for low-priority execution (idle scheduling, nice, ionice)."""
    prefix = []
    if shutil.which("chrt"):
        prefix += ["chrt", "--idle", "0"]
    if shutil.which("nice"):
        prefix += ["nice", "-n19"]
    if shutil.which("ionice"):
        prefix += ["ionice", "--class", "idle"]
    return prefix

LOW_PRIO = _build_low_prio_prefix()


def _set_low_priority():
    """Set low priority for the current process (used as ProcessPoolExecutor initializer)."""
    try:
        os.nice(19)
    except OSError:
        pass
