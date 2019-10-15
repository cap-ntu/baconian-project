"""This file is used for specifying various schedules that evolve over
time throughout the execution of the algorithm, such as:
 - learning rate for the optimizer
 - exploration epsilon for the epsilon greedy exploration strategy
 - beta parameter for beta parameter in prioritized replay

Each schedule has a function `value(t)` which returns the current value
of the parameter given the timestep t of the optimization procedure.
"""
from typeguard import typechecked
from baconian.common.error import *
from baconian.common.logging import ConsoleLogger


class Scheduler(object):
    def __init__(self):
        self.final_p = None
        self.initial_p = None

    def value(self):
        """Value of the schedule at time t"""
        raise NotImplementedError()


class ConstantScheduler(Scheduler):
    def __init__(self, value):
        """Value remains constant over time.

        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self._v = value
        Scheduler.__init__(self)

    def value(self):
        """See Schedule.value"""
        return self._v


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class PiecewiseScheduler(Scheduler):
    def __init__(self, endpoints, t_fn, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.

        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        Scheduler.__init__(self)

        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints
        self.t_fn = t_fn
        assert callable(self.t_fn)

    def value(self):
        """See Schedule.value"""
        t = wrap_t_fn(self.t_fn)
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


class LinearScheduler(Scheduler):
    @typechecked
    def __init__(self, t_fn, schedule_timesteps: int, final_p: float, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        Scheduler.__init__(self)
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p
        self.t_fn = t_fn
        if not callable(self.t_fn):
            raise TypeError("t_fn {} is not callable".format(self.t_fn))

    def value(self):
        t = wrap_t_fn(self.t_fn)
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class EventScheduler(Scheduler):
    def value(self) -> bool:
        return False


class PeriodicalEventSchedule(EventScheduler):
    """
    Trigger an event with certain scheduled period
    """

    def __init__(self, t_fn, trigger_every_step, after_t=0):
        super().__init__()
        self.t_fn = t_fn
        self.trigger_every_step = trigger_every_step
        self.after_t = after_t
        self.last_saved_t_value = -1

    def value(self) -> bool:
        """
        return a boolean, true for trigger this event, false for not.
        :return:
        """
        t = wrap_t_fn(self.t_fn)
        if t < self.after_t:
            return False
        else:
            if t - self.last_saved_t_value >= self.trigger_every_step:
                self.last_saved_t_value = t
                return True
            else:
                return False


def wrap_t_fn(t_fn):
    try:
        return t_fn()
    except StatusInfoNotRegisteredError:
        ConsoleLogger().print('error', 'StatusInfoNotRegisteredError occurred, return with 0')
        return 0
