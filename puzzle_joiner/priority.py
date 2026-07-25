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


def _for_all_threads(fn):
    """Call fn(tid) for every thread in the current process."""
    task_dir = "/proc/self/task"
    if os.path.isdir(task_dir):
        for tid_str in os.listdir(task_dir):
            try:
                fn(int(tid_str))
            except (OSError, AttributeError, ValueError):
                pass


def _set_process_low_priority():
    """Set all threads of the current process to nice 19 + SCHED_IDLE."""
    def _lower(tid):
        os.setpriority(os.PRIO_PROCESS, tid, 19)
        os.sched_setscheduler(tid, os.SCHED_IDLE, os.sched_param(0))
    _for_all_threads(_lower)


def _restore_process_priority():
    """Restore all threads to SCHED_OTHER. Nice revert requires CAP_SYS_NICE."""
    def _restore(tid):
        os.sched_setscheduler(tid, os.SCHED_OTHER, os.sched_param(0))
        os.setpriority(os.PRIO_PROCESS, tid, 0)
    _for_all_threads(_restore)
