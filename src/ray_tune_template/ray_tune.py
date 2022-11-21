from logging import getLogger
from typing import Optional, List, Dict

from aim.ext.resource.configs import DEFAULT_SYSTEM_TRACKING_INT
from aim.sdk.num_utils import is_number
from aim.sdk.run import Run

try:
    from ray import tune
    from ray.tune import Callback
    from ray.air import session
except ImportError:
    raise RuntimeError(
        'This contrib module requires Ray Tune to be installed. '
        'Please install it with command: \n pip install -U "ray[default]"'
    )

logger = getLogger(__name__)


class AimCallback(Callback):
    def __init__(self,
                 repo: Optional[str] = None,
                 experiment: Optional[str] = None,
                 system_tracking_interval: Optional[int]
                 = DEFAULT_SYSTEM_TRACKING_INT,
                 log_system_params: bool = True,
                 metrics: Optional[str] = None
                 ):
        self._repo_path = repo
        self._experiment_name = experiment
        self._system_tracking_interval = system_tracking_interval
        self._log_system_params = log_system_params
        self._metrics = metrics
        self._run = None
        self._run_hash = None
        self._log_value_warned = False

    @property
    def experiment(self):
        if not self._run:
            self.setup()
        return self._run

    def setup(
            self,
            stop: Optional["Stopper"] = None,
            num_samples: Optional[int] = None,
            total_num_samples: Optional[int] = None,
            **info,
    ):
        # ToDo add proper use of the Stopper
        if stop:
            return

        if not self._run:
            if self._run_hash:
                self._run = Run(
                    self._run_hash,
                    repo=self._repo_path,
                    system_tracking_interval=self._system_tracking_interval,
                )
            else:
                self._run = Run(
                    repo=self._repo_path,
                    experiment=self._experiment_name,
                    system_tracking_interval=self._system_tracking_interval,
                    log_system_params=self._log_system_params,
                )
                self._run_hash = self._run.hash

    def on_trial_result(
            self,
            iteration: int,
            trials: List["Trial"],
            trial: "Trial",
            result: Dict,
            **info,
    ):
        """
        Called after receiving a result from trial.
        Args:
            iteration:
            trials:
            trial:
            result:
            **info:

        Returns:

        """

        context  = None
        if "context" in result.keys():
            context = result["context"]

        if self._metrics:
            for metric in self._metrics:
                try:
                    self.experiment.track(value=result[metric], epoch=iteration, name=metric, context=context)
                except KeyError:
                    logger.warning(f"The metric {metric} is specified but not reported.")

    def on_trial_start(
            self, iteration: int, trials: List["Trial"], trial: "Trial", **info
    ):
        pass

    def on_trial_complete(
            self, iteration: int, trials: List["Trial"], trial: "Trial", **info
    ):
        pass

    def close(self):
        if self._run:
            self._run.close()
            del self._run
            self._run = None

    def __del__(self):
        self.close()
