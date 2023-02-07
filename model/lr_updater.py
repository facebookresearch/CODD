from mmcv.runner import HOOKS, LrUpdaterHook
import mmcv

@HOOKS.register_module()
class MultiGammaLrUpdaterHook(LrUpdaterHook):
    """Step LR scheduler.

    Args:
        step (list[int]): Step to decay the LR. If an int value is given,
            regard it as the decay interval. If a list is given, decay LR at
            these steps.
        gamma (list[float]): LR change ratios at certain steps.
    """

    def __init__(self, step, gamma, **kwargs):
        assert mmcv.is_list_of(step, int)
        assert mmcv.is_list_of(gamma, float)
        assert len(gamma) == len(step)
        assert all([s > 0 for s in step])
        self.step = step
        self.gamma = gamma
        super(MultiGammaLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter

        # calculate exponential term
        gamma = 1
        for i, s in enumerate(self.step):
            if progress < s:
                break
            gamma *= self.gamma[i]

        return base_lr * gamma
