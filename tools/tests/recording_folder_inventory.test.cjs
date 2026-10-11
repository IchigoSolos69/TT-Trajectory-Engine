'use strict';
// Isolated source-byte fixtures only. No decoder, models, Downloads or real reviews.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const fsp = fs.promises;
const path = require('node:path');
const crypto = require('node:crypto');
const {inventory, mergedLocations, writeLocations, videoFiles, parseInventoryArgs, compilationHashesFromManifest, loadMixedMatchManifest} = require('../recording_folder_inventory.cjs');
const artifacts = path.resolve(__dirname, '../../artifacts');
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const phone = 'phone_0123456789abcdef_1';

async function fixture(t) {
  await fsp.mkdir(artifacts, {recursive: true});
  const directory = await fsp.mkdtemp(path.join(artifacts, 'recording-folder-tests-'));
  assert.equal(path.dirname(directory), artifacts);
  t.after(async () => {
    assert.equal(path.dirname(directory), artifacts);
    assert.ok(path.basename(directory).startsWith('recording-folder-tests-'));
    await fsp.rm(directory, {recursive: true, force: true});
  });
  const downloads = path.join(directory, 'Downloads');
  const captures = path.join(directory, 'captures');
  for (const role of ['training', 'validation', 'seal']) await fsp.mkdir(path.join(downloads, role), {recursive: true});
  await fsp.mkdir(captures);
  return {directory, downloads, captures};
}
async function write(directory, relative, bytes) {
  const file = path.join(directory, relative);
  await fsp.mkdir(path.dirname(file), {recursive: true});
  await fsp.writeFile(file, bytes);
  return file;
}
async function json(directory, relative, value) { return write(directory, relative, JSON.stringify(value, null, 2) + '\n'); }
async function capture(captures, name, bytes, session = 'iphone_same_venue_session_1') {
  await json(captures, name + '/capture.json', {original_sha256: sha(bytes), original_path: path.join(captures, 'old source.mov'), session_id: session});
  return write(captures, name + '/annotations.json', 'unchanged annotation draft\n');
}

test('validated CLI parsing preserves the Downloads default and supports an arbitrary recordings root', () => {
  const defaultRoot = path.resolve('fixture home', 'Downloads');
  const defaults = parseInventoryArgs(['new report.json'], {defaultRecordingsRoot: defaultRoot});
  assert.equal(defaults.recordingsRoot, defaultRoot);
  assert.equal(defaults.destination, path.resolve('new report.json'));
  assert.equal(defaults.updateSourceLocations, false);
  const selected = parseInventoryArgs(['new report.json', '--recordings-root', 'python_tools/dataset/raw_videos', '--update-source-locations', '--mixed-match-manifest', 'old confirmed report.json'], {defaultRecordingsRoot: defaultRoot});
  assert.equal(selected.recordingsRoot, path.resolve('python_tools/dataset/raw_videos'));
  assert.equal(selected.mixedMatchManifest, path.resolve('old confirmed report.json'));
  assert.equal(selected.updateSourceLocations, true);
  assert.equal(selected.mixedMatchHighlights, false);
  assert.equal(parseInventoryArgs(['report.json', '--mixed-match-highlights']).mixedMatchHighlights, true);
});

test('CLI rejects missing values, unknown or repeated flags and contradictory compilation options', () => {
  for (const args of [[], ['--recordings-root', 'data'], ['r.json', '--recordings-root'], ['r.json', '--mixed-match-manifest'], ['r.json', '--recordings-root', '--update-source-locations'], ['r.json', '--mixed-match-manifest', ''], ['r.json', '--junk'], ['r.json', '--update-source-locations', '--update-source-locations'], ['r.json', '--recordings-root', 'a', '--recordings-root', 'b']]) {
    assert.throws(() => parseInventoryArgs(args), /Usage:/);
  }
  assert.throws(() => parseInventoryArgs(['r.json', '--mixed-match-highlights', '--mixed-match-manifest', 'm.json']), /mutually exclusive/);
});

test('compilation manifest admits only explicit true SHA identities and deduplicates repeated sources', () => {
  const confirmed = sha('known mixed compilation'), original = sha('established full match');
  const hashes = compilationHashesFromManifest(JSON.stringify({schema_version: 1, rows: [{sha256: confirmed, mixed_match_highlights: true, path: 'old unavailable filename.mp4'}, {sha256: confirmed, mixed_match_highlights: true}, {sha256: original, mixed_match_highlights: false}, {sha256: sha('unknown')}]}));
  assert.deepEqual([...hashes], [confirmed]);
  assert.equal(hashes.has(original), false);
});

test('malformed compilation declarations fail before inventory rather than implying a global classification', () => {
  for (const value of ['{broken', 'null', JSON.stringify({schema_version: 2, rows: []}), JSON.stringify({schema_version: 1}), JSON.stringify({schema_version: 1, rows: [null]}), JSON.stringify({schema_version: 1, rows: [{sha256: 'not a hash', mixed_match_highlights: true}]}), JSON.stringify({schema_version: 1, rows: [{sha256: sha('x'), mixed_match_highlights: 'true'}]})]) {
    assert.throws(() => compilationHashesFromManifest(value), /manifest/i);
  }
  assert.throws(() => compilationHashesFromManifest(JSON.stringify({schema_version: 1, rows: [{sha256: sha('x'), mixed_match_highlights: true}, {sha256: sha('x'), mixed_match_highlights: false}]})), /Conflicting/);
});

test('manifest loading binds exact bytes and canonical file path without opening any named video paths', async t => {
  const f = await fixture(t), confirmed = sha('compiled footage');
  const file = await json(f.directory, 'confirmed historical audit.json', {schema_version: 1, rows: [{sha256: confirmed, mixed_match_highlights: true, path: path.join(f.directory, 'nonexistent.mp4')}]});
  const bytes = await fsp.readFile(file), loaded = await loadMixedMatchManifest(file);
  assert.deepEqual([...loaded.hashes], [confirmed]);
  assert.deepEqual(loaded.provenance, {path: path.resolve(file), sha256: sha(bytes), confirmed_hash_count: 1});
  assert.deepEqual(await fsp.readFile(file), bytes);
});

test('arbitrary recordings root classifies relocated confirmed bytes while leaving new full-match sources unclassified', async t => {
  const f = await fixture(t), recordingsRoot = path.join(f.directory, 'dataset', 'raw_videos');
  for (const role of ['training', 'validation', 'seal']) await fsp.mkdir(path.join(recordingsRoot, role), {recursive: true});
  await write(recordingsRoot, 'validation/world_cup/renamed compilation.mp4', 'historical highlights');
  for (const name of ['game_1', 'game_2', 'test_1', 'test_2']) await write(recordingsRoot, 'training/world_cup/' + name + '.mp4', name + ' complete source bytes');
  // A misleading same name with different bytes is a new source, not a known compilation.
  await write(recordingsRoot, 'seal/world_cup/old compilation name.mp4', 'new complete source');
  await write(f.downloads, 'training/world_cup/ignored default.mp4', 'default should not be scanned');
  const result = await inventory({...f, recordingsRoot, mixedMatchHashes: new Set([sha('historical highlights')])});
  assert.equal(result.rows.length, 6);
  const compiled = result.rows.filter(row => row.mixed_match_highlights);
  assert.equal(compiled.length, 1);
  assert.equal(compiled[0].sha256, sha('historical highlights'));
  assert.equal(compiled[0].relative_path, path.join('validation', 'world_cup', 'renamed compilation.mp4'));
  assert.ok(compiled[0].reason_codes.includes('MIXED_MATCH_COMPILATION_NOT_INDEPENDENT'));
  for (const row of result.rows.filter(row => !row.mixed_match_highlights)) assert.equal(row.reason_codes.includes('MIXED_MATCH_COMPILATION_NOT_INDEPENDENT'), false);
  assert.equal(result.rows.every(row => row.independent_release_evidence === false), true);
});

test('inventory rejects invalid or competing direct compilation identity configurations', async t => {
  const f = await fixture(t);
  await assert.rejects(inventory({...f, mixedMatchHashes: ['not a Set']}), /Invalid confirmed/);
  await assert.rejects(inventory({...f, mixedMatchHashes: new Set(['invalid hash'])}), /Invalid confirmed/);
  await assert.rejects(inventory({...f, mixedMatchHighlights: true, mixedMatchHashes: new Set()}), /Choose either/);
});

test('recursively inventories Unicode and space paths without treating non-videos as footage', async t => {
  const f = await fixture(t);
  const actual = await write(f.downloads, 'training/iphone/家庭 camera/球 clip.MOV', 'phone bytes');
  const match = await write(f.downloads, 'validation/world_cup/nested match/clip one.mp4', 'match bytes');
  await write(f.downloads, 'training/iphone/notes.txt', 'not a recording');
  const calls = [];
  const result = await inventory({...f, onFile: row => calls.push(row)});
  assert.equal(result.rows.length, 2);
  assert.deepEqual(result.rows.map(row => row.path), [actual, match]);
  assert.deepEqual(result.rows.map(row => [row.requested_role, row.folder_declared_domain]), [['train', 'iphone'], ['val', 'match']]);
  assert.equal(result.rows[0].sha256, sha('phone bytes'));
  assert.deepEqual(await videoFiles(path.join(f.downloads, 'training')), [actual]);
  assert.equal(calls.length, 2);
});

test('original-byte SHA matches moved footage while equal filenames with different bytes remain unknown', async t => {
  const f = await fixture(t);
  const labels = await capture(f.captures, phone, 'reviewed original');
  const before = await fsp.readFile(labels);
  const actual = await write(f.downloads, 'training/iphone/moved source.MOV', 'reviewed original');
  await write(f.downloads, 'validation/iphone/moved source.MOV', 'different source');
  const result = await inventory(f);
  assert.equal(result.rows[0].matching_imports[0].clip, phone);
  assert.equal(result.rows[1].matching_imports.length, 0);
  assert.deepEqual(result.sources[sha('reviewed original')].paths, [actual]);
  assert.equal(Object.hasOwn(result.sources, sha('different source')), false);
  assert.deepEqual(await fsp.readFile(labels), before);
});

test('same original iPhone session cannot become validation by folder movement', async t => {
  const f = await fixture(t);
  await capture(f.captures, phone, 'original session bytes');
  await write(f.downloads, 'validation/iphone/IMG.MOV', 'original session bytes');
  const {rows} = await inventory(f);
  assert.equal(rows[0].requested_role, 'val');
  assert.equal(rows[0].effective_role, 'train');
  assert.ok(rows[0].reason_codes.includes('SAME_IPHONE_SESSION_CROSS_FOLDER'));
  assert.equal(rows[0].independent_release_evidence, false);
});

test('mixed-match highlight compilations are excluded from independent validation and test', async t => {
  const f = await fixture(t);
  await write(f.downloads, 'validation/world_cup/compilation.mp4', 'highlight v');
  await write(f.downloads, 'seal/world_cup/compilation.mp4', 'highlight t');
  await write(f.downloads, 'training/iphone/gameplay.MOV', 'camera');
  const {rows} = await inventory({...f, mixedMatchHighlights: true});
  for (const row of rows.filter(row => row.folder_declared_domain === 'match')) {
    assert.equal(row.mixed_match_highlights, true);
    assert.equal(row.effective_role, 'unassigned');
    assert.equal(row.independent_release_evidence, false);
    assert.ok(row.reason_codes.includes('MIXED_MATCH_COMPILATION_NOT_INDEPENDENT'));
  }
  assert.equal(rows.find(row => row.folder_declared_domain === 'iphone').mixed_match_highlights, false);
});

test('previously exposed original bytes in seal are flagged even when renamed', async t => {
  const f = await fixture(t);
  const bytes = 'previously evaluated original';
  await write(f.downloads, 'seal/world_cup/new name.mp4', bytes);
  const evidence = [{report: 'already-saved-metadata.json', report_sha256: sha('metadata')}];
  const {rows} = await inventory({...f, priorExposure: new Map([[sha(bytes), evidence]])});
  assert.deepEqual(rows[0].prior_development_evidence, evidence);
  assert.ok(rows[0].reason_codes.includes('PREVIOUSLY_EXPOSED_SOURCE_NOT_BLIND_TEST'));
  assert.equal(rows[0].independent_release_evidence, false);
});

test('exact cross-partition duplicates are identified by bytes and all requested roles', async t => {
  const f = await fixture(t);
  await write(f.downloads, 'training/world_cup/name A.mp4', 'same encoded bytes');
  await write(f.downloads, 'seal/world_cup/name B.mp4', 'same encoded bytes');
  const result = await inventory(f);
  assert.equal(result.exact_cross_split_duplicates.length, 1);
  assert.equal(result.exact_cross_split_duplicates[0].sha256, sha('same encoded bytes'));
  assert.deepEqual(result.exact_cross_split_duplicates[0].requested_roles, ['train', 'test']);
});

test('sealed registry identity is usable without opening poisoned C metadata or annotations', async t => {
  const f = await fixture(t);
  const sealed = 'phone_fedcba9876543210_1';
  const bytes = 'sealed original bytes';
  await json(f.captures, 'training_sessions.json', {schema_version: 1, sessions: {C: {role: 'test', sealed: true}}, captures: {[sealed]: {session_id: 'C', source_sha256: sha(bytes)}}});
  const poisoned = await write(f.captures, sealed + '/capture.json', 'INVALID JSON: must not be opened');
  await write(f.captures, sealed + '/annotations.json', 'INVALID JSON: must not be opened');
  const before = await fsp.readFile(poisoned);
  await write(f.downloads, 'training/iphone/renamed.MOV', bytes);
  const {rows} = await inventory(f);
  assert.equal(rows[0].matching_imports[0].sealed, true);
  assert.ok(rows[0].reason_codes.includes('SEALED_SOURCE_DEVELOPMENT_ALIAS'));
  assert.deepEqual(await fsp.readFile(poisoned), before);
});

test('unassigned capture junction to C is rejected before poisoned metadata can be read', async t => {
  const f = await fixture(t);
  const sealed = 'phone_fedcba9876543210_1';
  await json(f.captures, 'training_sessions.json', {schema_version: 1, sessions: {C: {role: 'test', sealed: true}}, captures: {[sealed]: {session_id: 'C', source_sha256: sha('sealed original')}}});
  const sealedDirectory = path.join(f.captures, sealed);
  await write(f.captures, sealed + '/capture.json', 'INVALID JSON: sealed target must not be read');
  await fsp.symlink(sealedDirectory, path.join(f.captures, phone), process.platform === 'win32' ? 'junction' : 'dir');
  await assert.rejects(inventory(f), /alias|junction|symlink/i);
});

test('recording folder junctions are rejected without descending into their targets', async t => {
  const f = await fixture(t);
  const outside = path.join(f.directory, 'outside videos');
  await fsp.mkdir(outside);
  await write(outside, 'secret.mp4', 'not inventoried');
  await fsp.symlink(outside, path.join(f.downloads, 'training/alias'), process.platform === 'win32' ? 'junction' : 'dir');
  await assert.rejects(videoFiles(path.join(f.downloads, 'training')), /alias|junction|symlink/i);
});

test('resolver merge prioritizes newly verified paths and preserves old lineage without mutation', async t => {
  const f = await fixture(t);
  const hash = sha('old source'), untouched = sha('another source');
  const oldPath = path.join(f.directory, 'old.mov');
  const newPath = path.join(f.directory, 'nested', 'new.mov');
  const previous = {schema_version: 1, note: 'retain manifest note', sources: {[hash]: {paths: [oldPath, newPath.toUpperCase()], model: 'kept'}, [untouched]: {paths: [path.join(f.directory, 'other.mov')]}}};
  const snapshot = structuredClone(previous);
  const merged = mergedLocations(previous, {[hash]: {paths: [newPath], reverified_at: 'fixture clock'}});
  assert.deepEqual(merged.sources[hash].paths, [newPath, oldPath]);
  assert.equal(merged.sources[hash].model, 'kept');
  assert.equal(merged.note, previous.note);
  assert.deepEqual(merged.sources[untouched], previous.sources[untouched]);
  assert.deepEqual(previous, snapshot);
  assert.throws(() => mergedLocations({schema_version: 2, sources: {}}, {}), /Invalid existing/);
});

test('audited migration retains exact old resolver bytes and leaves annotation drafts unchanged', async t => {
  const f = await fixture(t);
  const hash = sha('original');
  const oldPath = path.join(f.directory, 'old source.mov');
  const newPath = path.join(f.directory, 'new source.mov');
  const old = Buffer.from(JSON.stringify({schema_version: 1, sources: {[hash]: {paths: [oldPath]}}}, null, 4) + '\r\n');
  const resolver = await write(f.captures, 'source_locations.json', old);
  const annotation = await write(f.captures, phone + '/annotations.json', 'precious unsaved draft\n');
  const annotationBefore = await fsp.readFile(annotation);
  const backup = path.join(f.directory, 'exact old bytes.json');
  const migrated = await writeLocations({captures: f.captures, sources: {[hash]: {paths: [newPath]}}, backup, update: true});
  assert.deepEqual(await fsp.readFile(backup), old);
  assert.equal(migrated.previous_sha256, sha(old));
  assert.equal(migrated.sha256, sha(await fsp.readFile(resolver)));
  assert.deepEqual(JSON.parse(await fsp.readFile(resolver)).sources[hash].paths, [newPath, oldPath]);
  assert.deepEqual(await fsp.readFile(annotation), annotationBefore);
  await assert.rejects(fsp.access(path.join(f.captures, '.source_locations.migration.lock')), {code: 'ENOENT'});
});

test('existing resolver cannot be overwritten without explicit audited update', async t => {
  const f = await fixture(t);
  const old = Buffer.from('{"schema_version":1,"sources":{}}\n');
  const resolver = await write(f.captures, 'source_locations.json', old);
  const backup = path.join(f.directory, 'forbidden backup.json');
  await assert.rejects(writeLocations({captures: f.captures, sources: {}, backup}), /specify --update-source-locations/);
  assert.deepEqual(await fsp.readFile(resolver), old);
  await assert.rejects(fsp.access(backup), {code: 'ENOENT'});
});

test('exclusive migration lock blocks concurrent writer and preserves the existing resolver', async t => {
  const f = await fixture(t);
  const old = Buffer.from('{"schema_version":1,"sources":{}}\n');
  const resolver = await write(f.captures, 'source_locations.json', old);
  const lockFile = path.join(f.captures, '.source_locations.migration.lock');
  const owner = await fsp.open(lockFile, 'wx');
  try {
    await assert.rejects(writeLocations({captures: f.captures, sources: {}, backup: path.join(f.directory, 'backup.json'), update: true}), {code: 'EEXIST'});
    assert.deepEqual(await fsp.readFile(resolver), old);
    await fsp.access(lockFile); // Rejected contender cannot delete another writer's lock.
  } finally { await owner.close(); await fsp.unlink(lockFile); }
});

test('an existing backup cannot be overwritten and its failure preserves resolver and releases the lock', async t => {
  const f = await fixture(t);
  const old = Buffer.from('{"schema_version":1,"sources":{}}\n');
  const resolver = await write(f.captures, 'source_locations.json', old);
  const precious = Buffer.from('backup from a prior migration\n');
  const backup = await write(f.directory, 'existing backup.json', precious);
  await assert.rejects(writeLocations({captures: f.captures, sources: {}, backup, update: true}), {code: 'EEXIST'});
  assert.deepEqual(await fsp.readFile(resolver), old);
  assert.deepEqual(await fsp.readFile(backup), precious);
  await assert.rejects(fsp.access(path.join(f.captures, '.source_locations.migration.lock')), {code: 'ENOENT'});
});

test('rename failure preserves old resolver and backup, releases owned lock and retains a diagnostic candidate', async t => {
  const f = await fixture(t);
  const old = Buffer.from('{"schema_version":1,"sources":{}}\n');
  const resolver = await write(f.captures, 'source_locations.json', old);
  const backup = path.join(f.directory, 'backup.json');
  const rename = fsp.rename;
  fsp.rename = async (from, to) => {
    if (to === resolver) throw Object.assign(new Error('fixture atomic rename failure'), {code: 'EIO'});
    return rename.call(fsp, from, to);
  };
  try {
    await assert.rejects(writeLocations({captures: f.captures, sources: {}, backup, update: true}), /fixture atomic rename failure/);
  } finally { fsp.rename = rename; }
  assert.deepEqual(await fsp.readFile(resolver), old);
  assert.deepEqual(await fsp.readFile(backup), old);
  const remaining = await fsp.readdir(f.captures);
  assert.equal(remaining.includes('.source_locations.migration.lock'), false);
  const retained = remaining.filter(name => name.startsWith('source_locations.json.') && name.endsWith('.tmp'));
  assert.equal(retained.length, 1);
  assert.equal(JSON.parse(await fsp.readFile(path.join(f.captures, retained[0]))).schema_version, 1);
});
