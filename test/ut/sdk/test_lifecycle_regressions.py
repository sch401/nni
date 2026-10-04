"""Regressions for duplicate, delayed and failed trial lifecycle events."""
import pytest
import nni
from nni.algorithms.hpo.evolution_tuner import EvolutionTuner
from nni.algorithms.hpo.tpe_tuner import TpeTuner, create_liar
from nni.algorithms.hpo.pbt_tuner import PBTTuner
from nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner
from nni.algorithms.hpo.smac_tuner.smac_tuner import SMACTuner

SPACE = {'x': {'_type': 'uniform', '_value': [0.1, 1.0]}}

@pytest.mark.parametrize('success', [True, False])
@pytest.mark.parametrize('end_first', [True, False])
def test_evolution_orders(success, end_first):
    tuner = EvolutionTuner(population_size=1)
    tuner.update_search_space(SPACE)
    sent = []
    params = tuner.generate_multiple_parameters([0, 1, 2], st_callback=lambda *args: sent.append(args))[0]
    if end_first:
        tuner.trial_end(0, success)
    tuner.receive_trial_result(0, params, 0.0)
    tuner.receive_trial_result(0, params, 0.0)
    tuner.trial_end(0, success)
    tuner.trial_end(0, success)
    assert [entry[0] for entry in sent] == [1]
    tuner.receive_trial_result(1, sent[0][1], 1.0)
    tuner.trial_end(1, True)
    assert [entry[0] for entry in sent] == [1, 2]
    assert tuner.credit == 0
    assert tuner.num_running_trials == 1

def test_evolution_36_pending():
    tuner = EvolutionTuner(population_size=36)
    tuner.update_search_space(SPACE)
    sent = []
    params = tuner.generate_multiple_parameters(list(range(60)), st_callback=lambda *args: sent.append(args))
    assert len(params) == 36
    for i in range(24):
        tuner.receive_trial_result(i, params[i], i)
        tuner.trial_end(i, True)
    assert [entry[0] for entry in sent] == list(range(36, 60))
    assert tuner.num_running_trials == 36

@pytest.mark.parametrize('success', [True, False])
def test_tpe_end_before_final(success):
    tuner = TpeTuner()
    tuner.update_search_space(SPACE)
    params = tuner.generate_parameters(0)
    tuner.trial_end(0, success)
    tuner.receive_trial_result(0, params, 0.0)
    tuner.receive_trial_result(0, params, 0.0)
    assert len(next(iter(tuner._history.values()))) == int(success)

def test_tpe_none_liar():
    assert create_liar('none') is None

@pytest.mark.parametrize('end_first', [True, False])
def test_pbt_duplicate_failure(tmp_path, end_first):
    tuner = PBTTuner(population_size=2, all_checkpoint_dir=str(tmp_path))
    tuner.update_search_space(SPACE)
    params = tuner.generate_parameters(0)
    if end_first:
        tuner.trial_end(0, False)
    tuner.receive_trial_result(0, params, 0.0)
    tuner.receive_trial_result(0, params, 0.0)
    tuner.trial_end(0, False)
    tuner.trial_end(0, False)
    assert tuner.finished_trials == 1

def test_pbt_zero_import(tmp_path):
    tuner = PBTTuner(population_size=2, all_checkpoint_dir=str(tmp_path))
    tuner.update_search_space(SPACE)
    records = []
    for i, value in enumerate([0.0, -1.0]):
        checkpoint = tmp_path / str(i) / '0'
        checkpoint.mkdir(parents=True)
        records.append({'parameter': {'x': 0.5, 'save_checkpoint_dir': str(checkpoint)}, 'value': value})
    assert tuner.import_data(records) == 1
    assert any(trial.score == 0.0 for trial in tuner.population)

@pytest.mark.parametrize('tuner_class', [HyperoptTuner, SMACTuner])
def test_modern_optional_tuners(tuner_class):
    tuner = tuner_class('tpe') if tuner_class is HyperoptTuner else tuner_class()
    tuner.update_search_space(SPACE)
    params = tuner.generate_parameters(0)
    tuner.receive_trial_result(0, params, 0.0)
    data = [{'parameter': {'x': 0.4}, 'value': 0.0}]
    tuner.import_data(data)
    assert data[0]['parameter'] == {'x': 0.4}
    assert 0.1 <= tuner.generate_parameters(1)['x'] <= 1.0

def test_dispatcher_pending_preserves_first_parameters(monkeypatch):
    from nni.runtime import msg_dispatcher as module
    from nni.common.serializer import dump
    monkeypatch.setattr(module, '_next_parameter_id', 0)
    monkeypatch.setattr(module, '_trial_params', {})
    monkeypatch.setattr(module, '_ended_trials', set())
    tuner = EvolutionTuner(population_size=1)
    tuner.update_search_space(SPACE)
    dispatcher = module.MsgDispatcher('ws://_unittest_placeholder_', tuner)
    messages = []
    monkeypatch.setattr(dispatcher, 'send', lambda *args: messages.append(args))
    dispatcher.handle_request_trial_jobs(3)
    original = module._trial_params[0].copy()
    assert original and isinstance(original, dict)
    metric = {'parameter_id': 0, 'type': 'FINAL', 'value': dump(0.0), 'trial_job_id': 'trial-0'}
    dispatcher.handle_report_metric_data(metric.copy())
    dispatcher.handle_report_metric_data(metric.copy())
    end = {'hyper_params': dump({'parameter_id': 0}), 'event': 'SUCCEEDED', 'trial_job_id': 'trial-0'}
    dispatcher.handle_trial_end(end)
    dispatcher.handle_trial_end(end)
    assert module._trial_params[0] == original
    assert 1 in module._trial_params
    assert tuner.num_running_trials == 1

def test_experiment_registry_lock_with_filelock4(tmp_path):
    from nni.tools.nnictl.common_utils import get_file_lock
    lock = get_file_lock(str(tmp_path / 'registry'), stale=2)
    with lock.acquire(timeout=1):
        assert lock.is_locked
    assert not lock.is_locked
    assert not list(tmp_path.glob('registry.lock.*'))
