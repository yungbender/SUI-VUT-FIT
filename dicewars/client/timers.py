import signal
import logging


class EnterableTimerMixin:
    def __init__(self):
        raise NotImplementedError

    def __enter__(self):
        logging.debug('Starting clock with {:.3f}s left'.format(self.time_left))
        return self.run()

    def __exit__(self, type, value, traceback):
        self.stop()
        logging.debug('Stopped clock, now with {:.3f}s left'.format(self.time_left))


class FischerTimer(EnterableTimerMixin):
    def __init__(self, start_time, increment):
        self._time_left = start_time
        self._increment = increment

    def run(self):
        signal.setitimer(signal.ITIMER_REAL, self._time_left, 0)
        return self.time_left

    def stop(self):
        self._time_left, _ = signal.setitimer(signal.ITIMER_REAL, 0.0, 0)
        if self._time_left < 0:  # there may have been delay since raising the timeout
            self._time_left = 0.0
        self._time_left += self._increment

    @property
    def time_left(self):
        return self._time_left


class FixedTimer(EnterableTimerMixin):
    def __init__(self, start_time):
        self._time_left = start_time

    def run(self):
        signal.setitimer(signal.ITIMER_REAL, self._time_left, 0)
        return self.time_left

    def stop(self):
        self._time_left, _ = signal.setitimer(signal.ITIMER_REAL, 0.0, 0)
        if self._time_left < 0:  # there may have been delay since raising the timeout
            self._time_left = 0.0

    @property
    def time_left(self):
        return self._time_left
