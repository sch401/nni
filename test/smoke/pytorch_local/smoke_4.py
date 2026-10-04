"""Run a small local Evolution experiment with four concurrent processes."""
import json
import os
from pathlib import Path
import socket
import sys
import time

import nni
import torch
from nni.experiment import Experiment, RunMode

root = Path(__file__).resolve().parent
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
code = root / 'smoke_trial'
code.mkdir(exist_ok=True)
(code / 'trial.py').write_text('''import os, time
import nni
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
p = nni.get_next_parameter()
torch.manual_seed(7)
x = torch.randn(64, 4)
y = x.sum(dim=1, keepdim=True)
model = torch.nn.Linear(4, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001 + p['x'] / 100000)
for step in range(20):
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    if step in (9, 19):
        nni.report_intermediate_result(float(loss.detach()))
time.sleep(1 + int(os.environ['NNI_TRIAL_SEQ_ID']) % 4 / 10)
nni.report_final_result({'default': float(loss.detach()), 'parameter_x': p['x']})
''', encoding='utf-8')

experiment = Experiment('local')
experiment.config.experiment_name = 'Python 3.14 wheel regression'
experiment.config.trial_command = f'"{sys.executable}" trial.py'
experiment.config.trial_code_directory = code
experiment.config.experiment_working_directory = root / 'smoke_experiments'
experiment.config.search_space = {'x': {'_type': 'randint', '_value': [0, 10000]}}
experiment.config.trial_concurrency = 4
experiment.config.max_trial_number = 60
experiment.config.tuner.name = 'Evolution'
experiment.config.tuner.class_args = {'population_size': 4, 'optimize_mode': 'minimize'}
experiment.config.training_service.use_active_gpu = False
experiment.config.log_level = 'debug'
with socket.socket() as listener:
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
summary = {'python': sys.version, 'nni': nni.__version__, 'torch': torch.__version__, 'experiment_id': experiment.id,
           'concurrency': 4, 'max_trials': 60, 'port': port}
print(json.dumps(summary), flush=True)
try:
    experiment.start(port, run_mode=RunMode.Detach)
    deadline = time.monotonic() + 240
    last = None
    while time.monotonic() < deadline:
        status = experiment.get_status()
        jobs = experiment.list_trial_jobs()
        counts = {}
        for job in jobs:
            counts[job.status] = counts.get(job.status, 0) + 1
        progress = (status, counts)
        if progress != last:
            print(status, counts, flush=True)
            last = progress
        if status == 'ERROR':
            raise RuntimeError('Experiment entered ERROR')
        if status == 'DONE':
            break
        time.sleep(1)
    else:
        raise TimeoutError('Experiment did not finish within four minutes')
    results = experiment.export_data()
    assert len(results) == 60, len(results)
    assert counts == {'SUCCEEDED': 60}, counts
    for result in results:
        assert int(result.value['parameter_x']) == int(result.parameter['x']), result
        assert 0 <= float(result.value['default']) < 100, result
    summary.update(status=status, trial_counts=counts, exported_results=len(results),
                   parameter_metric_mismatches=0)
    (root / 'smoke_4_result.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('PASSED: 60 trials and parameter/metric pairing', flush=True)
finally:
    experiment.stop()
