'use strict';
// Development-only manifests from isolated byte fixtures; no training or real footage.
const test = require('node:test');
const assert = require('node:assert/strict');
const fsp = require('node:fs').promises;
const path = require('node:path');
const crypto = require('node:crypto');
const {spawnSync} = require('node:child_process');
const artifacts = path.resolve(__dirname, '../../artifacts');
const builder = path.resolve(__dirname, '../prepare_development_folder_split.cjs');
const digest = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

async function fixture(t) {
  await fsp.mkdir(artifacts, {recursive: true});
  const directory = await fsp.mkdtemp(path.join(artifacts, 'development-split-tests-'));
  assert.equal(path.dirname(directory), artifacts);
  t.after(async () => {
    assert.equal(path.dirname(directory), artifacts);
    assert.ok(path.basename(directory).startsWith('development-split-tests-'));
    await fsp.rm(directory, {recursive: true, force: true});
  });
  return {directory, input: path.join(directory, 'audited sources.json'), output: path.join(directory, 'development only.json')};
}
async function source(f, name, bytes = 'fixture encoded bytes', extra = {}) {
  const file = path.join(f.directory, name);
  await fsp.mkdir(path.dirname(file), {recursive: true});
  await fsp.writeFile(file, bytes);
  const stat = await fsp.stat(file);
  return {path: file, sha256: digest(bytes), bytes: stat.size, source_mtime_ms: stat.mtimeMs,
    requested_role: 'train', matching_imports: [], known_session_ids: [], folder_declared_domain: null,
    effective_role: 'unassigned', mixed_match_highlights: false, prior_development_evidence: [],
    reason_codes: [], reason: 'Fixture source needs review', independent_release_evidence: false, ...extra};
}
async function audit(f, rows, extra = {}) {
  const bytes = Buffer.from(JSON.stringify({schema_version: 1, rows, production_labels_modified: false,
    dataset_partitions_modified: false, mixed_match_highlights_declared: false, ...extra}, null, 2) + '\n');
  await fsp.writeFile(f.input, bytes);
  return bytes;
}
async function prepare(f, options = {}) {
  const helper = require('../prepare_development_folder_split.cjs');
  assert.equal(typeof helper.prepare, 'function', 'Builder exposes its tested asynchronous API');
  await helper.prepare({input: f.input, output: f.output, ...options});
  return JSON.parse(await fsp.readFile(f.output));
}

test('nested Unicode and space source paths remain development-only with exact audit identity', async t => {
  const f = await fixture(t);
  const row = await source(f, 'training/iphone/球 camera clip.MOV', 'phone bytes', {folder_declared_domain: 'iphone'});
  const bytes = await audit(f, [row]);
  const report = await prepare(f);
  assert.equal(report.purpose, 'DEVELOPMENT_CLIP_HOLDOUT_ONLY');
  assert.equal(report.inventory_sha256, digest(bytes));
  assert.equal(report.sources[0].path, row.path);
  assert.equal(report.sources[0].source_mtime_ms, row.source_mtime_ms);
  assert.equal(report.independent_release_evidence, false);
  assert.equal(report.production_registry_modified, false);
  assert.equal(report.sealed_test_labels_consumed, false);
  assert.equal(report.contains_automatically_reviewed_ground_truth, false);
  assert.equal(report.freshness_checked_by, 'STREAM_SHA256_AND_STAT');
});

test('known original iPhone session wins over mistaken world_cup placement and public-match option', async t => {
  const f = await fixture(t);
  const row = await source(f, 'validation/world_cup/original.MOV', 'original', {
    requested_role: 'val', folder_declared_domain: 'match', effective_role: 'train',
    matching_imports: [{clip: 'phone_0123456789abcdef_1', session_id: 'iphone_same_venue_session_1'}],
    known_session_ids: ['iphone_same_venue_session_1'], reason_codes: ['SAME_IPHONE_SESSION_CROSS_FOLDER', 'FOLDER_DOMAIN_CONFLICTS_WITH_KNOWN_SESSION']});
  await audit(f, [row]);
  const item = (await prepare(f, {publicMatchSources: true})).sources[0];
  assert.equal(item.declared_domain, 'iphone');
  assert.equal(item.folder_declared_domain, 'match');
  assert.equal(item.requested_split, 'val');
  assert.equal(item.effective_role, 'train');
  assert.deepEqual(item.reason_codes, row.reason_codes);
});

test('explicit unmatched iphone folder is never classified as match by the compatibility option', async t => {
  const f = await fixture(t);
  await audit(f, [await source(f, 'training/iphone/new camera.MOV', 'new camera', {folder_declared_domain: 'iphone'})]);
  const report = await prepare(f, {publicMatchSources: true});
  assert.equal(report.sources[0].declared_domain, 'iphone');
  assert.equal(report.sources[0].effective_role, 'unassigned');
});

test('legacy flat unknown source can retain explicit user public-match declaration without pretending a known session', async t => {
  const f = await fixture(t);
  await audit(f, [await source(f, 'flat public.mp4', 'public')]);
  const report = await prepare(f, {publicMatchSources: true});
  assert.equal(report.sources[0].declared_domain, 'match');
  assert.equal(report.sources[0].session_group_status, 'UNASSIGNED');
  assert.equal(report.sources[0].independent_release_evidence, false);
});

test('mixed-match compilation exclusion and reason survive builder migration unchanged', async t => {
  const f = await fixture(t);
  const row = await source(f, 'seal/world_cup/highlights.mp4', 'highlight bytes', {requested_role: 'test',
    folder_declared_domain: 'match', mixed_match_highlights: true, effective_role: 'unassigned',
    reason_codes: ['MIXED_MATCH_COMPILATION_NOT_INDEPENDENT'], reason: 'Reviewed source grouping remains required'});
  await audit(f, [row], {mixed_match_highlights_declared: true});
  const item = (await prepare(f)).sources[0];
  assert.equal(item.mixed_match_highlights, true);
  assert.equal(item.effective_role, 'unassigned');
  assert.equal(item.independent_release_evidence, false);
  assert.deepEqual(item.reason_codes, row.reason_codes);
  assert.equal(item.reason, row.reason);
});

test('previous development exposure remains attached to requested sealed clip instead of becoming blind evidence', async t => {
  const f = await fixture(t);
  const evidence = [{report: 'old numerical parity.json', report_sha256: digest('report bytes')}];
  const row = await source(f, 'seal/world_cup/renamed demo.mp4', 'previous demo', {requested_role: 'test',
    folder_declared_domain: 'match', prior_development_evidence: evidence, reason_codes: ['PREVIOUSLY_EXPOSED_SOURCE_NOT_BLIND_TEST']});
  await audit(f, [row]);
  const item = (await prepare(f)).sources[0];
  assert.deepEqual(item.prior_development_evidence, evidence);
  assert.deepEqual(item.reason_codes, row.reason_codes);
  assert.equal(item.independent_release_evidence, false);
});

test('exact cross-split duplicates are grouped by hash rather than filenames', async t => {
  const f = await fixture(t);
  const train = await source(f, 'training/A.mp4', 'identical bytes');
  const seal = await source(f, 'seal/B.mp4', 'identical bytes', {requested_role: 'test'});
  const val = await source(f, 'validation/A.mp4', 'different bytes', {requested_role: 'val'});
  await audit(f, [train, seal, val]);
  const report = await prepare(f);
  assert.deepEqual(report.exact_cross_split_duplicates, [{sha256: train.sha256, splits: ['test', 'train']}]);
  assert.deepEqual(report.folder_counts, {train: 1, val: 1, test: 1});
});

test('one known session split across requested folders remains explicitly reported', async t => {
  const f = await fixture(t);
  const known = {known_session_ids: ['iphone_same_venue_session_1'], effective_role: 'train'};
  await audit(f, [await source(f, 'training/one.MOV', 'first', known),
    await source(f, 'validation/two.MOV', 'second', {...known, requested_role: 'val'})]);
  const report = await prepare(f);
  assert.deepEqual(report.shared_sessions, [{session_id: 'iphone_same_venue_session_1', splits: ['train', 'val']}]);
  assert.deepEqual(report.sources.map(row => row.effective_role), ['train', 'train']);
});

test('same-size source replacement fails stream hash check even without recorded mtime', async t => {
  const f = await fixture(t);
  const row = await source(f, 'training/source.mp4', 'original');
  delete row.source_mtime_ms; // Legacy inventory still requires immutable byte identity.
  await audit(f, [row]);
  await fsp.writeFile(row.path, 'mutated!');
  assert.equal((await fsp.stat(row.path)).size, row.bytes);
  await assert.rejects(prepare(f), /moved|changed|hash|identity|SHA/i);
  await assert.rejects(fsp.access(f.output), {code: 'ENOENT'});
});

test('unchanged legacy inventory without recorded mtime is rehashed before use', async t => {
  const f = await fixture(t);
  const row = await source(f, 'legacy original.mp4');
  delete row.source_mtime_ms;
  await audit(f, [row]);
  const report = await prepare(f);
  assert.equal(report.sources[0].sha256, row.sha256);
  assert.equal(report.freshness_checked_by, 'STREAM_SHA256_AND_STAT');
});

test('recorded mtime drift fails even when encoded bytes and size are unchanged', async t => {
  const f = await fixture(t);
  const row = await source(f, 'unchanged bytes.mp4');
  await audit(f, [row]);
  const original = await fsp.stat(row.path);
  await fsp.utimes(row.path, original.atime, new Date(original.mtimeMs + 5000));
  await assert.rejects(prepare(f), /moved|changed|mtime|audit/i);
  await assert.rejects(fsp.access(f.output), {code: 'ENOENT'});
});

test('same-content folder junction replacement is rejected despite matching SHA, size and mtime', async t => {
  const f = await fixture(t);
  const row = await source(f, 'training/iphone/original.MOV', 'unchanged original bytes', {folder_declared_domain: 'iphone'});
  await audit(f, [row]);
  const auditedFolder = path.dirname(row.path);
  const relocatedFolder = path.join(f.directory, 'real original folder');
  await fsp.rename(auditedFolder, relocatedFolder);
  await fsp.symlink(relocatedFolder, auditedFolder, process.platform === 'win32' ? 'junction' : 'dir');
  const aliasedStat = await fsp.stat(row.path);
  assert.equal(aliasedStat.size, row.bytes);
  assert.equal(aliasedStat.mtimeMs, row.source_mtime_ms);
  assert.equal(digest(await fsp.readFile(row.path)), row.sha256);
  await assert.rejects(prepare(f), /alias|junction|symlink|path changed/i);
  await assert.rejects(fsp.access(f.output), {code: 'ENOENT'});
});

test('missing or size-changed audited footage refuses to publish output', async t => {
  const f = await fixture(t);
  const row = await source(f, 'missing source.mp4');
  await audit(f, [row]);
  await fsp.unlink(row.path);
  await assert.rejects(prepare(f));
  await assert.rejects(fsp.access(f.output), {code: 'ENOENT'});
  await fsp.writeFile(row.path, 'different and longer original');
  await assert.rejects(prepare(f), /moved|changed|size|audit/i);
});

test('malformed or unaudited inventories cannot produce a manifest', async t => {
  const f = await fixture(t);
  const row = await source(f, 'source.mp4');
  for (const invalid of [{schema_version: 2}, {production_labels_modified: true}, {rows: {}},
    {rows: [{...row, sha256: 'not a hash'}]}, {rows: [{...row, requested_role: 'made_up'}]}, {rows: [{...row, path: 'relative.mp4'}]}]) {
    await audit(f, [row], invalid);
    await assert.rejects(prepare(f));
    await assert.rejects(fsp.access(f.output), {code: 'ENOENT'});
  }
});

test('existing output cannot be overwritten and original input remains byte-identical', async t => {
  const f = await fixture(t);
  const inputBytes = await audit(f, [await source(f, 'source.mp4')]);
  const previous = Buffer.from('previous reviewed manifest\n');
  await fsp.writeFile(f.output, previous);
  await assert.rejects(prepare(f), /exists|NEW_MANIFEST|Usage|new output|overwrite/i);
  assert.deepEqual(await fsp.readFile(f.output), previous);
  assert.deepEqual(await fsp.readFile(f.input), inputBytes);
});

test('CLI validates options and produces the same safe manifest through its supported interface', async t => {
  const f = await fixture(t);
  await audit(f, [await source(f, 'iphone/球 file.MOV', 'new phone', {folder_declared_domain: 'iphone'})]);
  const invalid = spawnSync(process.execPath, [builder, f.input, f.output, '--pretend-independent'], {encoding: 'utf8', windowsHide: true});
  assert.notEqual(invalid.status, 0);
  assert.match(invalid.stderr, /Unknown|option|Usage/i);
  await assert.rejects(fsp.access(f.output), {code: 'ENOENT'});
  const result = spawnSync(process.execPath, [builder, f.input, f.output, '--public-match-sources'], {encoding: 'utf8', windowsHide: true});
  assert.equal(result.status, 0, result.stderr);
  const report = JSON.parse(await fsp.readFile(f.output));
  assert.equal(report.sources[0].declared_domain, 'iphone');
  assert.equal(report.independent_release_evidence, false);
});
