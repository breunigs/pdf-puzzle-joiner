from PySide6.QtCore import QObject, Signal, QRunnable

from .priority import _set_process_low_priority, _restore_process_priority


class WorkerSignals(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(str)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        _set_process_low_priority()
        try:
            result = self.fn(*self.args, **self.kwargs, progress=self.signals.progress)
            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            _restore_process_priority()
