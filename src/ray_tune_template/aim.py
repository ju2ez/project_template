import logging
import pathlib

import numpy as np

from typing import TYPE_CHECKING, Dict, Optional

from aim.ext.resource import DEFAULT_SYSTEM_TRACKING_INT
from ray.tune.logger.logger import Logger, LoggerCallback
from ray.util.debug import log_once
from ray.tune.result import (
    TRAINING_ITERATION,
    TIME_TOTAL_S,
    TIMESTEPS_TOTAL,
)
from ray.tune.utils import flatten_dict
from ray.util.annotations import PublicAPI

if TYPE_CHECKING:
    from ray.tune.experiment.trial import Trial  # noqa: F401

logger = logging.getLogger(__name__)

VALID_SUMMARY_TYPES = [int, float, np.float32, np.float64, np.int32, np.int64]


@PublicAPI
class AimLogger(Logger):
    """Aim Logger.
    """

    VALID_HPARAMS = (str, bool, int, float, list, type(None))
    VALID_NP_HPARAMS = (np.bool8, np.float32, np.float64, np.int32, np.int64)

    def __init__(self, config: Dict, logdir: str, trial: Optional["Trial"] = None):
        super().__init__(config, logdir, trial)
        self.last_result = None

    def _init(self):
        try:
            from aim.sdk.run import Run
            from aim import Image
            from aim import Distribution
            from aim.ext.resource.configs import DEFAULT_SYSTEM_TRACKING_INT

        except ImportError:
            if log_once("aim-install"):
                logger.info('pip install aim to enable logging with aim.')
            raise

        self._run = Run(repo=self.logdir, experiment=self.trial.experiment_tag)
        self.last_result = None

    def on_result(self, result: Dict):
        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        tmp = result.copy()
        for k in ["config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION]:
            if k in tmp:
                del tmp[k]  # not useful to log these

        context = None
        if context in tmp:
            context = tmp["context"]
            del tmp["context"]

        flat_result = flatten_dict(tmp, delimiter="/")
        path = ["ray", "tune"]
        valid_result = {}

        for attr, value in flat_result.items():
            full_attr = "/".join(path + [attr])
            if isinstance(value, tuple(VALID_SUMMARY_TYPES)) and not np.isnan(value):
                valid_result[full_attr] = value
                self._run.track(value=value, name=full_attr, step=step, context=context)
            elif (isinstance(value, list) and len(value) > 0) or (
                    isinstance(value, np.ndarray) and value.size > 0
            ):
                valid_result[full_attr] = value

                # Must be an RGB image
                if isinstance(value, np.ndarray) and value.ndim == 3:
                    self._run.track(value=Image(value, attr), step=step, context=context)
                    continue

                # Todo: add functoinality of figures, distributions,..
                try:
                    self._file_writer.add_histogram(full_attr, value, global_step=step)
                # In case TensorboardX still doesn't think it's a valid value
                # (e.g. `[[]]`), warn and move on.
                except (ValueError, TypeError):
                    if log_once("invalid_tbx_value"):
                        logger.warning(
                            "You are trying to log an invalid value ({}={}) "
                            "via {}!".format(full_attr, value, type(self).__name__)
                        )

        self.last_result = valid_result

    def close(self):
        if self._run is not None:
            if self.trial and self.trial.evaluated_params and self.last_result:
                flat_result = flatten_dict(self.last_result, delimiter="/")
                scrubbed_result = {
                    k: value
                    for k, value in flat_result.items()
                    if isinstance(value, tuple(VALID_SUMMARY_TYPES))
                }
                self._try_log_hparams(scrubbed_result)
            self._run.close()

    def _try_log_hparams(self, result):
        # log hyperparameters
        flat_params = flatten_dict(self.trial.evaluated_params)
        scrubbed_params = {
            k: v for k, v in flat_params.items() if isinstance(v, self.VALID_HPARAMS)
        }

        np_params = {
            k: v.tolist()
            for k, v in flat_params.items()
            if isinstance(v, self.VALID_NP_HPARAMS)
        }

        scrubbed_params.update(np_params)

        removed = {
            k: v
            for k, v in flat_params.items()
            if not isinstance(v, self.VALID_HPARAMS + self.VALID_NP_HPARAMS)
        }
        if removed:
            logger.info(
                "Removed the following hyperparameter values when "
                "logging to tensorboard: %s",
                str(removed),
            )

        try:
            self._run["hparams"] = scrubbed_params
        except Exception:
            logger.exception(
                "Aim failed to log hparams. "
                "This may be due to an unsupported type "
                "in the hyperparameter values."
            )


@PublicAPI
class AimCallback(LoggerCallback):
    """Aim Logger.
    Logs the aim results.
    """

    VALID_HPARAMS = (str, bool, int, float, list, type(None))
    VALID_NP_HPARAMS = (np.bool8, np.float32, np.float64, np.int32, np.int64)

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
        self._log_value_warned = False

        try:
            from aim.sdk import Run
            self._run_cls = Run
            from aim import Image
            from aim import Distribution
            from aim.ext.resource.configs import DEFAULT_SYSTEM_TRACKING_INT

        except ImportError:
            if log_once("aim-install"):
                logger.info('pip install aim to see aim files.')
            raise

        self._trial_run: Dict["Trial", Run] = {}
        self._trial_result: Dict["Trial", Dict] = {}

    def _create_run(self):
        """
        Creates a aim Run instance
        Returns: Run

        """
        run = self._run_cls(
                repo=self._repo_path,
                experiment=self._experiment_name,
                system_tracking_interval=self._system_tracking_interval,
                log_system_params=self._log_system_params,
            )
        return run

    def log_trial_start(self, trial: "Trial"):
        trial.init_logdir()
        self._trial_run[trial] = self._create_run()
        self._trial_run[trial].add_tag(trial.trial_id)
        self._trial_result[trial] = {}

    def log_trial_result(self, iteration: int, trial: "Trial", result: Dict):
        if trial not in self._trial_run:
            self.log_trial_start(trial)

        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        tmp = result.copy()
        for k in ["config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION]:
            if k in tmp:
                del tmp[k]  # not useful to log these

        context = None
        if "context" in tmp:
            context = tmp["context"]
            del tmp["context"]

        epoch = None
        if "epoch" in tmp:
            epoch = tmp["epoch"]
            del tmp["epoch"]

        if self._metrics:
            for metric in self._metrics:
                try:
                    self.experiment.track(value=result[metric], epoch=iteration, name=metric, context=context)
                except KeyError:
                    logger.warning(f"The metric {metric} is specified but not reported.")

        else:
            # if no metric is specified log everything that is reported
            flat_result = flatten_dict(tmp, delimiter="/")
            valid_result = {}

            for attr, value in flat_result.items():
                if isinstance(value, tuple(VALID_SUMMARY_TYPES)) and not np.isnan(value):
                    valid_result[attr] = value
                    self._trial_run[trial].track(value=value, name=attr, epoch=epoch, step=step, context=context)
                elif (isinstance(value, list) and len(value) > 0) or (
                        isinstance(value, np.ndarray) and value.size > 0
                ):
                    valid_result[attr] = value

                    # Must be video
                    if isinstance(value, np.ndarray) and value.ndim == 5:
                        self._trial_writer[trial].add_video(
                            attr, value, global_step=step, fps=20
                        )
                        continue

                    try:
                        self._trial_writer[trial].add_histogram(
                            attr, value, global_step=step
                        )
                    # In case TensorboardX still doesn't think it's a valid value
                    # (e.g. `[[]]`), warn and move on.
                    except (ValueError, TypeError):
                        if log_once("invalid_tbx_value"):
                            logger.warning(
                                "You are trying to log an invalid value ({}={}) "
                                "via {}!".format(attr, value, type(self).__name__)
                            )
            self._trial_result[trial] = valid_result

    def log_trial_end(self, trial: "Trial", failed: bool = False):
        if trial in self._trial_run:
            if trial and trial.evaluated_params and self._trial_result[trial]:
                flat_result = flatten_dict(self._trial_result[trial], delimiter="/")
                scrubbed_result = {
                    k: value
                    for k, value in flat_result.items()
                    if isinstance(value, tuple(VALID_SUMMARY_TYPES))
                }
                self._try_log_hparams(trial, scrubbed_result)
            self._trial_run[trial].close()
            del self._trial_run[trial]
            del self._trial_result[trial]

    def _try_log_hparams(self, trial: "Trial", result: Dict):
        flat_params = flatten_dict(trial.evaluated_params)
        scrubbed_params = {
            k: v for k, v in flat_params.items() if isinstance(v, self.VALID_HPARAMS)
        }

        np_params = {
            k: v.tolist()
            for k, v in flat_params.items()
            if isinstance(v, self.VALID_NP_HPARAMS)
        }

        scrubbed_params.update(np_params)

        removed = {
            k: v
            for k, v in flat_params.items()
            if not isinstance(v, self.VALID_HPARAMS + self.VALID_NP_HPARAMS)
        }
        if removed:
            logger.info(
                "Removed the following hyperparameter values when "
                "logging to aim: %s",
                str(removed),
            )

        try:
            self._trial_run[trial]['hparams'] = scrubbed_params
        except Exception:
            logger.exception(
                "aim failed to log hparams. "
                "This may be due to an unsupported type "
                "in the hyperparameter values."
            )
