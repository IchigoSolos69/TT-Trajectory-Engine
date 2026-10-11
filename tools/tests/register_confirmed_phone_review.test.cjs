'use strict';
// Real production assignment rules with isolated byte fixtures, never real labels.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const fsp = fs.promises;
const path = require('node:path');
const {registerConfirmedReview, parseArgs, CLIPS, SESSION, hash} = require('../register_confirmed_phone_review.cjs');
const artifacts = path.resolve(__dirname, '../../artifacts');

async function fixture(t) {
  const directory = await fsp.mkdtemp(path.join(artifacts, 'confirmed-phone-tests-'));
  t.after(async () => {
    assert.equal(path.dirname(directory), artifacts);
    assert.ok(path.basename(directory).startsWith('confirmed-phone-tests-'));
    await fsp.rm(directory, {recursive: true, force: true});
  });
  const base = path.join(directory, 'captures'), source = path.join(directory, 'relocated', 'Video Project 6.mp4');
  await fsp.mkdir(base); await fsp.mkdir(path.dirname(source));
  const original = Buffer.from('fixture original recording bytes'), sha = hash(original);
  await fsp.writeFile(source, original);
  for (const clip of CLIPS) {
    const dir = path.join(base, clip); await fsp.mkdir(dir); await fsp.mkdir(path.join(dir, 'annotation_history'));
    await fsp.writeFile(path.join(dir, 'capture.json'), JSON.stringify({source: 'recorded_camera', capture_origin: 'unspecified', session_id: SESSION,
      original_path: 'C:\\former\\Video Project 6.mp4', original_sha256: sha, original_bytes: original.length, sampled_frames: 1}));
    await fsp.writeFile(path.join(dir, 'annotations.json'), '{"frame_1.jpg":{"reviewed":true,"ball":[[0.1,0.2,0.01,0.01]]}}\n');
    await fsp.writeFile(path.join(dir, 'frames.jsonl'), '{"file":"frame_1.jpg","source_frame":1,"elapsed_ms":0}\n');
    await fsp.writeFile(path.join(dir, 'frame_1.jpg'), Buffer.from('fixture saved review pixels ' + clip));
    await fsp.writeFile(path.join(dir, 'annotation_history', 'saved.json'), 'unchanged review revision\n');
  }
  const other = 'phone_aaaaaaaaaaaaaaaa_1', sealed = 'phone_cccccccccccccccc_1';
  await fsp.mkdir(path.join(base, other));
  await fsp.writeFile(path.join(base, other, 'capture.json'), JSON.stringify({source: 'recorded_camera', session_id: SESSION, original_sha256: 'a'.repeat(64)}));
  const registry = {schema_version: 1, sessions: {[SESSION]: {role: 'train', stage: 'existing', sealed: false, assigned_at: 'prior'},
    untouched_C: {role: 'test', stage: 'C', sealed: true, assigned_at: 'prior'}},
  captures: {[other]: {session_id: SESSION, domain: 'iphone', assigned_at: 'prior'},
    [sealed]: {session_id: 'untouched_C', domain: 'pc_camera', source_sha256: 'c'.repeat(64), assigned_at: 'prior'}}};
  const registryRaw = Buffer.from(JSON.stringify(registry, null, 3) + '\n'), registryFile = path.join(base, 'training_sessions.json');
  await fsp.writeFile(registryFile, registryRaw);
  const resolver = {schema_version: 1, sources: {[sha]: {paths: [source]}}};
  await fsp.writeFile(path.join(base, 'source_locations.json'), JSON.stringify(resolver));
  const output = path.join(directory, 'new audit.json');
  return {directory, base, source, sha, registry, registryRaw, registryFile, output, other, sealed};
}
const run = (f, extra = {}) => registerConfirmedReview({output: f.output, captures: f.base, expectedSourceSha: f.sha, ...extra});
async function captureEdit(f, patch, clip = CLIPS[0]) {
  const file = path.join(f.base, clip, 'capture.json');
  await fsp.writeFile(file, JSON.stringify({...JSON.parse(await fsp.readFile(file)), ...patch}));
}
async function registryEdit(f, mutate) {
  const value = JSON.parse(await fsp.readFile(f.registryFile)); mutate(value);
  const bytes = Buffer.from(JSON.stringify(value)); await fsp.writeFile(f.registryFile, bytes); return bytes;
}

test('default dry run keeps registry/reviews unchanged and binds the exact user confirmation', async t => {
  const f = await fixture(t), report = await run(f);
  assert.equal(report.status, 'DRY_RUN'); assert.equal(report.registry_modified, false);
  assert.equal(report.registry_before_sha256, hash(f.registryRaw));
  assert.deepEqual(report.assignments.map(value => value.clip), CLIPS);
  assert.match(report.user_confirmation, /both Video Project 6 review batches/);
  assert.equal(report.preserved_reverified, true); assert.equal(report.preserved_file_count, 10);
  assert.equal(report.sealed_test_files_opened, false);
  assert.ok((await fsp.readFile(f.registryFile)).equals(f.registryRaw));
  for (const [file, digest] of Object.entries(report.preserved_sha256)) assert.equal(hash(await fsp.readFile(file)), digest);
  assert.equal(fs.existsSync(path.join(f.base, '.training-sessions.lock')), false);
  assert.equal(fs.existsSync(path.join(f.base, 'training_session_history')), false);
});

test('apply registers exactly both batches, retains source identity and saves exact prior registry bytes', async t => {
  const f = await fixture(t), report = await run(f, {apply: true, expectedRegistrySha: hash(f.registryRaw)});
  assert.equal(report.status, 'APPLIED'); assert.equal(report.preserved_reverified, true);
  assert.ok((await fsp.readFile(report.registry_backup.path)).equals(f.registryRaw));
  const current = JSON.parse(await fsp.readFile(f.registryFile));
  assert.deepEqual(current.sessions, f.registry.sessions);
  assert.deepEqual(current.captures[f.other], f.registry.captures[f.other]);
  assert.deepEqual(current.captures[f.sealed], f.registry.captures[f.sealed]);
  assert.equal(Object.keys(current.captures).length, Object.keys(f.registry.captures).length + 2);
  for (const clip of CLIPS) {
    assert.equal(current.captures[clip].session_id, SESSION); assert.equal(current.captures[clip].domain, 'iphone');
    assert.equal(current.captures[clip].source_sha256, f.sha);
  }
  for (const [file, digest] of Object.entries(report.preserved_sha256)) assert.equal(hash(await fsp.readFile(file)), digest);
  assert.equal(fs.existsSync(path.join(f.base, '.training-sessions.lock')), false);
  await assert.rejects(run(f, {output: path.join(f.directory, 'second.json'), apply: true, expectedRegistrySha: report.registry_after_sha256}), /already assigned/);
});

test('CLI defaults dry run, rejects extra scope and requires a frozen registry SHA for apply', async t => {
  assert.equal(parseArgs(['audit.json']).apply, false);
  assert.throws(() => parseArgs(['audit.json', '--source-sha256', 'a'.repeat(64)]));
  assert.throws(() => parseArgs(['audit.json', '--apply', '--apply']));
  const f = await fixture(t); await assert.rejects(run(f, {apply: true}), /requires --expected-registry/);
  assert.ok((await fsp.readFile(f.registryFile)).equals(f.registryRaw));
});

test('capture source/session/name/SHA/size mismatches never register either batch', async t => {
  for (const patch of [{source: 'live'}, {session_id: 'invented_session'}, {original_path: 'other.mp4'},
    {original_sha256: 'd'.repeat(64)}, {original_bytes: 999}]) {
    const f = await fixture(t); await captureEdit(f, patch);
    await assert.rejects(run(f, {apply: true, expectedRegistrySha: hash(f.registryRaw)}), /mismatch|do not match/);
    assert.ok((await fsp.readFile(f.registryFile)).equals(f.registryRaw));
    assert.equal(fs.existsSync(path.join(f.base, '.training-sessions.lock')), false);
  }
});

test('capture directory aliases are rejected before reading reviewed evidence', async t => {
  const f = await fixture(t), capture = path.join(f.base, CLIPS[0]), real = path.join(f.directory, 'alias-target');
  await fsp.rename(capture, real); await fsp.symlink(real, capture, 'junction');
  await assert.rejects(run(f, {apply: true, expectedRegistrySha: hash(f.registryRaw)}), /Alias/);
  assert.ok((await fsp.readFile(f.registryFile)).equals(f.registryRaw));
});

test('relocated source aliases, missing resolver identity and changed source bytes are rejected', async t => {
  for (const kind of ['alias', 'missing', 'changed']) {
    const f = await fixture(t);
    if (kind === 'alias') {
      const real = path.join(f.directory, 'actual-original'); await fsp.rename(path.dirname(f.source), real);
      await fsp.symlink(real, path.dirname(f.source), 'junction');
    } else if (kind === 'missing') await fsp.writeFile(path.join(f.base, 'source_locations.json'), JSON.stringify({schema_version: 1, sources: {}}));
    else await fsp.writeFile(f.source, 'changed original');
    await assert.rejects(run(f, {apply: true, expectedRegistrySha: hash(f.registryRaw)}), /Alias|absent|do not match/);
    assert.ok((await fsp.readFile(f.registryFile)).equals(f.registryRaw));
  }
});

test('existing capture assignment and fixed original session role cannot be overridden', async t => {
  for (const kind of ['assigned', 'role']) {
    const f = await fixture(t), before = await registryEdit(f, value => {
      if (kind === 'assigned') value.captures[CLIPS[0]] = {session_id: SESSION, domain: 'pc_camera', assigned_at: 'prior'};
      else value.sessions[SESSION] = {stage: 'B', role: 'val', sealed: false, assigned_at: 'prior'};
    });
    await assert.rejects(run(f, {apply: true, expectedRegistrySha: hash(before)}), /already assigned|must already/);
    assert.ok((await fsp.readFile(f.registryFile)).equals(before));
  }
});

test('whole-source conflicts include legacy nonsealed metadata and sealed registry-only identities', async t => {
  for (const kind of ['legacy', 'sealed']) {
    const f = await fixture(t); let before = f.registryRaw;
    if (kind === 'legacy') {
      before = await registryEdit(f, value => { value.captures[f.other].domain = 'pc_camera'; });
      await captureEdit(f, {original_sha256: f.sha}, f.other);
    } else before = await registryEdit(f, value => { value.captures[f.sealed].source_sha256 = f.sha; });
    await assert.rejects(run(f, {apply: true, expectedRegistrySha: hash(before)}), /conflicting|One original recording/);
    assert.ok((await fsp.readFile(f.registryFile)).equals(before));
  }
});

test('preexisting lock and stale expected revision preserve other writers and their assignments', async t => {
  const f = await fixture(t), lock = path.join(f.base, '.training-sessions.lock'); await fsp.writeFile(lock, 'other writer');
  await assert.rejects(run(f, {apply: true, expectedRegistrySha: hash(f.registryRaw)}), /Another session/);
  assert.equal(await fsp.readFile(lock, 'utf8'), 'other writer');
  await fsp.unlink(lock);
  await assert.rejects(run(f, {apply: true, expectedRegistrySha: 'f'.repeat(64)}), /changed since dry run/);
  assert.ok((await fsp.readFile(f.registryFile)).equals(f.registryRaw));
});

test('concurrent registry or review changes abort before replacing the registry', async t => {
  for (const kind of ['registry', 'review']) {
    const f = await fixture(t), concurrentlyWritten = Buffer.from(JSON.stringify({...f.registry, concurrency_marker: true}));
    await assert.rejects(run(f, {apply: true, expectedRegistrySha: hash(f.registryRaw), hooks: {beforeReplace: async () => {
      if (kind === 'registry') await fsp.writeFile(f.registryFile, concurrentlyWritten);
      else await fsp.writeFile(path.join(f.base, CLIPS[0], 'annotations.json'), 'concurrently changed review');
    }}}), /changed during registration/);
    assert.ok((await fsp.readFile(f.registryFile)).equals(kind === 'registry' ? concurrentlyWritten : f.registryRaw));
    const audit = JSON.parse(await fsp.readFile(f.output)); assert.equal(audit.status, 'FAILED'); assert.equal(audit.registry_modified, false);
    assert.equal(fs.existsSync(path.join(f.base, '.training-sessions.lock')), false);
    assert.equal((await fsp.readdir(f.base)).some(name => name.endsWith('.sessions.tmp')), false);
  }
});

test('failed replacement and conflicting history bytes leave registry unchanged', async t => {
  for (const kind of ['replace', 'history']) {
    const f = await fixture(t);
    if (kind === 'history') {
      const history = path.join(f.base, 'training_session_history'); await fsp.mkdir(history);
      await fsp.writeFile(path.join(history, hash(f.registryRaw) + '.json'), 'incorrect prior bytes');
    }
    await assert.rejects(run(f, {apply: true, expectedRegistrySha: hash(f.registryRaw), hooks: kind === 'replace' ? {beforeReplace: async () => { throw Error('fixture replacement failure'); }} : {}}), /history bytes disagree|replacement failure/);
    assert.ok((await fsp.readFile(f.registryFile)).equals(f.registryRaw));
    assert.equal(fs.existsSync(path.join(f.base, '.training-sessions.lock')), false);
  }
});

test('post-commit verification failure clearly records the committed registry and exact backup', async t => {
  const f = await fixture(t);
  await assert.rejects(run(f, {apply: true, expectedRegistrySha: hash(f.registryRaw), hooks: {afterReplace: async () => {
    await fsp.writeFile(path.join(f.base, 'source_locations.json'), '{}');
  }}}), /resolver changed/);
  const current = JSON.parse(await fsp.readFile(f.registryFile)), audit = JSON.parse(await fsp.readFile(f.output));
  for (const clip of CLIPS) assert.equal(current.captures[clip].session_id, SESSION);
  assert.equal(audit.status, 'FAILED'); assert.equal(audit.registry_modified, true);
  assert.equal(audit.registry_committed_sha256, hash(await fsp.readFile(f.registryFile)));
  assert.ok((await fsp.readFile(audit.registry_backup.path)).equals(f.registryRaw));
  assert.equal(fs.existsSync(path.join(f.base, '.training-sessions.lock')), false);
});

test('audit paths cannot overwrite protected evidence or an earlier audit', async t => {
  const f = await fixture(t);
  await assert.rejects(run(f, {output: path.join(f.base, CLIPS[0], 'annotations.json')}), /outside the protected/);
  await fsp.writeFile(f.output, 'preserved earlier audit'); await assert.rejects(run(f), /Existing audit/);
  assert.equal(await fsp.readFile(f.output, 'utf8'), 'preserved earlier audit');
});
