# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
smac_tuner.py
"""

import logging
from copy import deepcopy
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
from schema import Schema, Optional

from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory.dataclasses import TrialInfo, TrialValue
from smac.runhistory.enumerations import StatusType
from ConfigSpace import Configuration, ConfigurationSpace, Categorical, Float, Integer, Constant

import nni
from nni import ClassArgsValidator
from nni.common.hpo_utils import validate_search_space
from nni.tuner import Tuner
from nni.utils import OptimizeMode, extract_scalar_reward


logger = logging.getLogger('smac_AutoML')

class SMACClassArgsValidator(ClassArgsValidator):
    def validate_class_args(self, **kwargs):
        Schema({
            'optimize_mode': self.choices('optimize_mode', 'maximize', 'minimize'),
            Optional('config_dedup'): bool
        }).validate(kwargs)

class SMACTuner(Tuner):
    """
    `SMAC <https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf>`__ is based on Sequential Model-Based Optimization (SMBO).
    It adapts the most prominent previously used model class (Gaussian stochastic process models)
    and introduces the model class of random forests to SMBO in order to handle categorical parameters.

    The SMAC supported by nni is a wrapper on `the SMAC3 github repo <https://github.com/automl/SMAC3>`__,
    following NNI tuner interface :class:`nni.tuner.Tuner`. For algorithm details of SMAC, please refer to the paper
    :footcite:t:`hutter2011sequential`.

    Note that SMAC on nni only supports a subset of the types in
    :doc:`search space </hpo/search_space>`:
    ``choice``, ``randint``, ``uniform``, ``loguniform``, and ``quniform``.

    Note that SMAC needs additional installation using the following command:

    .. code-block:: bash

        pip install nni[SMAC]

    This wrapper uses the public ask/tell API of SMAC 2.

    Examples
    --------

    .. code-block::

        config.tuner.name = 'SMAC'
        config.tuner.class_args = {
            'optimize_mode': 'maximize'
        }

    Parameters
    ----------
    optimize_mode : str
        Optimize mode, 'maximize' or 'minimize', by default 'maximize'
    config_dedup : bool
        If True, the tuner will not generate a configuration that has been already generated.
        If False, a configuration may be generated twice, but it is rare for relatively large search space.
    """

    def __init__(self, optimize_mode="maximize", config_dedup=False):
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.dedup = config_dedup
        self.total_data = {}
        self.categorical_dict = {}
        self._seen = set()
        self.optimizer = None
        self._output_dir = TemporaryDirectory(prefix="nni-smac-")

    def update_search_space(self, search_space):
        if self.optimizer is not None:
            raise RuntimeError("SMAC does not support changing an initialized search space")
        validate_search_space(search_space, ['choice', 'randint', 'uniform', 'quniform', 'loguniform'])
        self.cs = ConfigurationSpace(seed=0)
        for key, spec in search_space.items():
            kind, values = spec['_type'], spec['_value']
            if kind in ('choice', 'quniform'):
                if kind == 'quniform':
                    low, high, q = values
                    if q <= 0:
                        raise ValueError("quniform requires a positive step")
                    values = np.clip(np.arange(np.round(low / q), np.round(high / q) + 1) * q, low, high).tolist()
                self.categorical_dict[key] = deepcopy(values)
                hyperparameter = Categorical(key, list(range(len(values))))
            elif kind == 'randint':
                low, high = values
                hyperparameter = Constant(key, low) if high == low + 1 else Integer(key, (low, high - 1))
            else:
                low, high = values
                hyperparameter = Constant(key, low) if low == high else Float(key, (low, high), log=kind == 'loguniform')
            self.cs.add(hyperparameter)
        scenario = Scenario(self.cs, deterministic=True, n_trials=2147483647,
                            output_directory=Path(self._output_dir.name))
        self.optimizer = HyperparameterOptimizationFacade(scenario, overwrite=True, logging_level=False)

    def param_postprocess(self, challenger_dict):
        return {key: deepcopy(self.categorical_dict[key][int(value)]) if key in self.categorical_dict else value
                for key, value in challenger_dict.items()}

    def generate_parameters(self, parameter_id, **kwargs):
        info = self.optimizer.ask()
        config = dict(info.config)
        signature = tuple(sorted(config.items()))
        if self.dedup and signature in self._seen:
            raise nni.NoMoreTrialError("SMAC returned a configuration already requested")
        self._seen.add(signature)
        self.total_data[parameter_id] = info
        return self.param_postprocess(config)

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        info = self.total_data.pop(parameter_id, None)
        if info is None:
            return
        reward = extract_scalar_reward(value)
        if self.optimize_mode is OptimizeMode.Maximize:
            reward = -reward
        self.optimizer.tell(info, TrialValue(cost=reward), save=False)

    def trial_end(self, parameter_id, success, **kwargs):
        if not success:
            info = self.total_data.pop(parameter_id, None)
            if info is not None:
                self.optimizer.tell(info, TrialValue(cost=float('inf'), status=StatusType.CRASHED), save=False)

    def import_data(self, data):
        for record in data:
            if record['value'] is None:
                continue
            params = deepcopy(record['parameter'])
            for key, values in self.categorical_dict.items():
                params[key] = values.index(params[key])
            config = Configuration(self.cs, values=params)
            reward = extract_scalar_reward(record['value'])
            if self.optimize_mode is OptimizeMode.Maximize:
                reward = -reward
            self.optimizer.tell(TrialInfo(config=config, seed=0), TrialValue(cost=reward), save=False)
            self._seen.add(tuple(sorted(dict(config).items())))
