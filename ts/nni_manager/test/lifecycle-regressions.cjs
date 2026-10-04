// Run after `npm run build`: node --test test/lifecycle-regressions.cjs
const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { setTimeout: delay } = require('node:timers/promises');
require('app-module-path').addPath(path.resolve(__dirname, '../dist'));
const globals = require('common/globals').default;
Object.assign(globals, {args: {experimentId: 'lifecycle-test', logLevel: 'error'},
  paths: {experimentRoot: os.tmpdir()},
  rest: {urlJoin: (...parts) => parts.join('/'), registerWebSocketHandler() {}},
  logStream: {writeLineSync() {}}, shutdown: {register() {}}});

test('V3 parameters stay bound to their own trial when requested in reverse order', async () => {
  const callbacks = {};
  const sent = new Map();
  const created = [];
  const fake = {
    async init() {},
    onRequestParameter(fn) {callbacks.request = fn;}, onMetric() {}, onTrialStart() {},
    onTrialEnd(fn) {callbacks.end = fn;}, onEnvironmentUpdate() {},
    async start() {return [{id: 'local'}];}, async uploadDirectory() {},
    async createTrial(env, cmd, dir, seq, id) {created.push(id); return id;},
    async sendParameter(id, parameter) {sent.set(id, parameter);},
    async stopTrial(id) {await callbacks.end(id, Date.now(), 0);}
  };
  require('training_service/v3/factory').trainingServiceFactoryV3 = () => fake;
  const {V3asV1} = require('training_service/v3/compat');
  const service = new V3asV1({trialCommand: 'unused', trialCodeDirectory: '.', platform: 'local'});
  await service.start();
  const forms = [0, 1].map(i => ({sequenceId: i, hyperParameters: {value: JSON.stringify({parameter_id: i}), index: 0}}));
  const trials = await Promise.all(forms.map(form => service.submitTrialJob(form)));
  await callbacks.request(created[1]);
  await callbacks.request(created[0]);
  trials.forEach((trial, i) => assert.equal(sent.get(trial.id), forms[i].hyperParameters.value));
  await service.cancelTrialJob(trials[0].id, true);
  assert.equal(trials[0].status, 'EARLY_STOPPED');
});

test('late final metrics are persisted once and failed persistence can retry', async () => {
  const {NNIManager} = require('core/nnimanager');
  const manager = Object.create(NNIManager.prototype);
  const stored = [], forwarded = [];
  Object.assign(manager, {trialJobs: new Map(), endedTrialIds: new Set(['ended']),
    receivedFinalMetrics: new Set(), log: {debug() {}, warning() {}},
    dataStore: {async storeMetricData(...args) {stored.push(args);}},
    dispatcher: {sendCommand(...args) {forwarded.push(args);}}});
  const metric = {id: 'ended', data: JSON.stringify({parameter_id: 3, type: 'FINAL', value: '0.0'})};
  await Promise.all([manager.onTrialJobMetrics(metric), manager.onTrialJobMetrics(metric)]);
  assert.equal(stored.length, 1);
  assert.equal(forwarded.length, 1);
  const retry = {id: 'ended', data: JSON.stringify({parameter_id: 4, type: 'FINAL', value: '1.0'})};
  manager.dataStore.storeMetricData = async () => {throw new Error('simulated database failure');};
  await assert.rejects(manager.onTrialJobMetrics(retry));
  manager.dataStore.storeMetricData = async (...args) => stored.push(args);
  await manager.onTrialJobMetrics(retry);
  assert.equal(forwarded.length, 2);
});

test('export rejects mismatched parameter IDs and retains scalar zero', async () => {
  const {NNIDataStore} = require('core/nniDataStore');
  const store = Object.create(NNIDataStore.prototype);
  store.log = {warning() {}};
  store.listTrialJobs = async () => [
    {trialJobId: 'wrong', hyperParameters: [JSON.stringify({parameter_id: 1, parameters: {x: 1}})],
      finalMetricData: [{parameterId: '2', data: '0'}]},
    {trialJobId: 'right', hyperParameters: [JSON.stringify({parameter_id: 3, parameters: {x: 3}})],
      finalMetricData: [{parameterId: '3', data: '0'}]}
  ];
  const records = JSON.parse(await store.exportTrialHpConfigs());
  assert.deepEqual(records, [{parameter: {x: 3}, value: 0, trialJobId: 'right'}]);
});

test('killing a Windows trial terminates its child processes', {skip: process.platform !== 'win32', timeout: 20000}, async () => {
  const {TrialProcess} = require('common/trial_keeper/process');
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'nni-tree-test-'));
  const script = path.join(dir, 'parent.py');
  await fs.writeFile(script, "import subprocess, sys, time, pathlib\np = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\npathlib.Path('child.pid').write_text(str(p.pid))\ntime.sleep(120)\n");
  const proc = new TrialProcess('kill-tree');
  try {
    assert.equal(await proc.spawn({command: `"${process.env.NNI_TEST_PYTHON}" "${script}"`,
      codeDirectory: dir, outputDirectory: dir, commandChannelUrl: 'unused', platform: 'local', environmentVariables: {}}), true);
    const pidFile = path.join(dir, 'child.pid');
    let pid;
    for (let i = 0; i < 100; i++) {
      try {pid = Number(await fs.readFile(pidFile, 'utf8')); break;} catch {await delay(50);}
    }
    assert.ok(pid);
    await proc.kill();
    await delay(100);
    assert.throws(() => process.kill(pid, 0), {code: 'ESRCH'});
  } finally {await proc.kill();}
});
