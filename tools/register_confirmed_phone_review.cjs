'use strict';
// Register only the two already reviewed batches covered by the user's confirmation.
// Production assignment/source-conflict rules are reused; no labels are rewritten.
const fs = require('node:fs');
const fsp = fs.promises;
const path = require('node:path');
const crypto = require('node:crypto');
const vm = require('node:vm');

const ROOT = path.resolve(__dirname, '..');
const SESSION = 'iphone_same_venue_session_1';
const CLIPS = ['phone_452627e98a3c2652_1', 'phone_452627e98a3c2652_2'];
const SOURCE_SHA = '452627e98a3c2652bbfe08bdaff90a7e1a3c5c5d6d21a22e75d61fe14a6bf706';
const CONFIRMATION = 'On 2026-10-12 the user confirmed that both Video Project 6 review batches came entirely from the original iPhone gameplay session.';
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const equalPath = (a, b) => path.resolve(a).toLowerCase() === path.resolve(b).toLowerCase();
const inside = (base, file) => { const relative = path.relative(base, file); return !relative || (!relative.startsWith('..' + path.sep) && relative !== '..' && !path.isAbsolute(relative)); };

function productionSessions() {
  const ts = require('../frontend/node_modules/typescript');
  const load = (name, dependencies = {}) => {
    const file = path.join(ROOT, 'frontend', 'src', 'lib', name + '.ts');
    const scope = {exports: {}, require: dependency => {
      if (!(dependency in dependencies)) throw Error('Unexpected production dependency: ' + dependency);
      return dependencies[dependency];
    }};
    vm.runInNewContext(ts.transpileModule(fs.readFileSync(file, 'utf8'), {compilerOptions: {module: ts.ModuleKind.CommonJS}}).outputText, scope, {filename: file});
    return scope.exports;
  };
  const geometry = load('reviewGeometry');
  const annotations = load('reviewAnnotations', {'./reviewGeometry': geometry});
  const intervals = load('trainingAbsenceIntervals');
  return load('trainingSessions', {'./reviewAnnotations': annotations, './trainingAbsenceIntervals': intervals});
}

async function unaliased(file, directory = false) {
  const stat = await fsp.lstat(file);
  if (stat.isSymbolicLink() || (directory ? !stat.isDirectory() : !stat.isFile()) || !equalPath(await fsp.realpath(file), file)) throw Error('Alias or unexpected file type excluded: ' + file);
  return stat;
}
async function readOwned(file) { await unaliased(file); return fsp.readFile(file); }
async function streamHash(file) {
  const before = await unaliased(file);
  const digest = await new Promise((resolve, reject) => {
    const checksum = crypto.createHash('sha256');
    const stream = fs.createReadStream(file);
    stream.on('data', bytes => checksum.update(bytes));
    stream.on('error', reject);
    stream.on('end', () => resolve(checksum.digest('hex')));
  });
  const after = await unaliased(file);
  if (before.size !== after.size || before.mtimeMs !== after.mtimeMs) throw Error('Original changed during hashing');
  return {path: file, sha256: digest, bytes: after.size, mtime_ms: after.mtimeMs};
}
async function captureSnapshot(base) {
  const result = {};
  async function walk(directory) {
    await unaliased(directory, true);
    for (const entry of (await fsp.readdir(directory, {withFileTypes: true})).sort((a, b) => a.name.localeCompare(b.name))) {
      const file = path.join(directory, entry.name);
      if (entry.isSymbolicLink()) throw Error('Review evidence alias excluded: ' + file);
      if (entry.isDirectory()) await walk(file);
      else result[file] = hash(await readOwned(file));
    }
  }
  for (const clip of CLIPS) await walk(path.join(base, clip));
  return result;
}
async function writeNewSynced(file, bytes) {
  const handle = await fsp.open(file, 'wx');
  try { await handle.writeFile(bytes); await handle.sync(); } finally { await handle.close(); }
}

/** expectedSourceSha/hooks are fixture-only API inputs; the CLI pins the actual original SHA. */
async function registerConfirmedReview({output, captures = path.join(ROOT, 'training_data', 'live'), apply = false,
  expectedRegistrySha, expectedSourceSha = SOURCE_SHA, hooks = {}}) {
  const base = path.resolve(captures), destination = path.resolve(output || '');
  if (!output) throw Error('A new audit output is required');
  if (!/^[0-9a-f]{64}$/.test(expectedSourceSha) || (expectedRegistrySha !== undefined && !/^[0-9a-f]{64}$/.test(expectedRegistrySha))) throw Error('Invalid expected SHA256');
  if (apply && !expectedRegistrySha) throw Error('--apply requires --expected-registry-sha256 from the dry-run audit');
  await unaliased(base, true);
  if (inside(base, destination)) throw Error('Audit output must be outside the protected training directory');
  await unaliased(path.dirname(destination), true);
  try { await fsp.lstat(destination); throw Error('Existing audit output is preserved'); } catch (error) { if (error.code !== 'ENOENT') throw error; }
  const sessions = productionSessions(), registryFile = path.join(base, 'training_sessions.json');
  const resolverFile = path.join(base, 'source_locations.json'), lockPath = path.join(base, '.training-sessions.lock');
  let lock, temporary, auditHandle, report, committed = false;
  const publish = async () => {
    const bytes = Buffer.from(JSON.stringify(report, null, 2) + '\n');
    await auditHandle.write(bytes, 0, bytes.length, 0);
    await auditHandle.truncate(bytes.length);
    await auditHandle.sync();
  };
  try {
    if (apply) {
      try { lock = await fsp.open(lockPath, 'wx'); } catch (error) {
        if (error.code === 'EEXIST') throw Error('Another session assignment is saving; existing lock preserved');
        throw error;
      }
    }
    const registryRaw = await readOwned(registryFile), revision = hash(registryRaw);
    if (expectedRegistrySha && revision !== expectedRegistrySha) throw Error('Session assignments changed since dry run');
    const registry = sessions.validateTrainingRegistry(JSON.parse(registryRaw));
    const session = registry.sessions[SESSION];
    if (!session || session.stage !== 'existing' || session.role !== 'train' || session.sealed !== false) throw Error('The confirmed original session must already be existing/train and unsealed');
    for (const clip of CLIPS) if (registry.captures[clip]) throw Error('Capture already assigned; preserve the existing assignment: ' + clip);
    const manifests = {};
    for (const clip of CLIPS) {
      const directory = path.join(base, clip); await unaliased(directory, true);
      const capture = JSON.parse(await readOwned(path.join(directory, 'capture.json')));
      if (capture.source !== 'recorded_camera' || capture.session_id !== SESSION || capture.original_sha256 !== expectedSourceSha ||
          typeof capture.original_path !== 'string' || path.win32.basename(capture.original_path).toLowerCase() !== 'video project 6.mp4') throw Error('Confirmed capture source/session/identity mismatch: ' + clip);
      manifests[clip] = capture;
    }
    const now = new Date().toISOString();
    let next = registry;
    // Reject registry-bound sealed source aliases before any review files/original bytes.
    for (const clip of CLIPS) next = sessions.assignTrainingCapture(next, {clip, session_id: SESSION, stage: 'existing', domain: 'iphone'}, manifests[clip].session_id, now, expectedSourceSha);
    const resolverRaw = await readOwned(resolverFile), resolver = JSON.parse(resolverRaw);
    const locations = resolver.sources?.[expectedSourceSha]?.paths;
    if (resolver.schema_version !== 1 || !Array.isArray(locations) || !locations.length || locations.some(file => typeof file !== 'string' || !path.isAbsolute(file))) throw Error('Original identity is absent from the relocated source resolver');
    let original;
    for (const file of locations) {
      try { await fsp.lstat(file); } catch (error) { if (error.code === 'ENOENT') continue; throw error; }
      if (path.basename(file).toLowerCase() !== 'video project 6.mp4') throw Error('Relocated original filename mismatch');
      original = await streamHash(file); break;
    }
    if (!original || original.sha256 !== expectedSourceSha || CLIPS.some(clip => manifests[clip].original_bytes !== original.bytes)) throw Error('Relocated original bytes do not match both confirmed captures');
    const preserved = await captureSnapshot(base), metadata = {};
    const sourceMetadata = [];
    for (const [clip, assignment] of Object.entries(next.captures)) {
      if (next.sessions[assignment.session_id].sealed) continue; // C files are never opened.
      const directory = path.join(base, clip); await unaliased(directory, true);
      const file = path.join(directory, 'capture.json'), raw = await readOwned(file);
      metadata[file] = hash(raw);
      const capture = JSON.parse(raw);
      if (capture.session_id && capture.session_id !== assignment.session_id) throw Error('Assigned original session identity mismatch: ' + clip);
      sourceMetadata.push({clip, original_sha256: capture.original_sha256});
    }
    if (sessions.wholeSourceConflicts(next, sourceMetadata).size) throw Error('Original source has conflicting session, role or device assignments');
    sessions.validateTrainingRegistry(next);
    const serialized = Buffer.from(JSON.stringify(next, null, 2) + '\n');
    const verify = async (registryDigest = revision) => {
      if (hash(await readOwned(registryFile)) !== registryDigest) throw Error('Session assignments changed during registration');
      if (hash(await readOwned(resolverFile)) !== hash(resolverRaw)) throw Error('Source resolver changed during registration');
      if (JSON.stringify(await captureSnapshot(base)) !== JSON.stringify(preserved)) throw Error('Reviewed files changed during registration');
      for (const [file, digest] of Object.entries(metadata)) if (hash(await readOwned(file)) !== digest) throw Error('Capture provenance changed during registration');
      if ((await streamHash(original.path)).sha256 !== expectedSourceSha) throw Error('Original source bytes changed during registration');
    };
    report = {schema_version: 1, scope: 'Only user-confirmed Video Project 6 batches registered as existing iPhone training data',
      user_confirmation: CONFIRMATION, mode: apply ? 'apply' : 'dry_run', status: 'PLANNED', registry_path: registryFile,
      registry_before_sha256: revision, registry_proposed_sha256: hash(serialized), source_locations_sha256: hash(resolverRaw),
      original, assignments: CLIPS.map(clip => ({clip, ...next.captures[clip]})), preserved_sha256: preserved,
      preserved_file_count: Object.keys(preserved).length, production_assignment_helper: 'frontend/src/lib/trainingSessions.ts:assignTrainingCapture',
      new_sessions_created: false, labels_modified: false, source_manifests_modified: false, sealed_test_files_opened: false,
      training_started: false, registry_modified: false, preserved_reverified: false};
    auditHandle = await fsp.open(destination, 'wx'); await publish();
    if (hooks.beforeCommit) await hooks.beforeCommit({base, registryFile, resolverFile});
    await verify();
    if (apply) {
      const history = path.join(base, 'training_session_history');
      await fsp.mkdir(history, {recursive: true}); await unaliased(history, true);
      const backup = path.join(history, revision + '.json');
      try { await writeNewSynced(backup, registryRaw); } catch (error) {
        if (error.code !== 'EEXIST') throw error;
        if (!(await readOwned(backup)).equals(registryRaw)) throw Error('Existing registry history bytes disagree');
      }
      report.registry_backup = {path: backup, sha256: revision, exact_original_bytes: true};
      temporary = path.join(base, crypto.randomUUID() + '.sessions.tmp');
      await writeNewSynced(temporary, serialized);
      if (hooks.beforeReplace) await hooks.beforeReplace({base, registryFile, temporary});
      await verify();
      await fsp.rename(temporary, registryFile); temporary = undefined; committed = true;
      report.registry_modified = true; report.registry_committed_sha256 = hash(serialized);
      if (hooks.afterReplace) await hooks.afterReplace({base, registryFile});
      await verify(hash(serialized)); report.registry_after_sha256 = hash(serialized);
    }
    if (JSON.stringify(await captureSnapshot(base)) !== JSON.stringify(preserved)) throw Error('Reviewed files changed after registration');
    report.status = apply ? 'APPLIED' : 'DRY_RUN'; report.preserved_reverified = true; await publish();
    return report;
  } catch (error) {
    if (report && auditHandle) { report.status = 'FAILED'; report.registry_modified = committed; report.error = error.message; await publish(); }
    throw error;
  } finally {
    if (auditHandle) await auditHandle.close();
    if (temporary) await fsp.unlink(temporary).catch(error => { if (error.code !== 'ENOENT') throw error; });
    if (lock) { await lock.close(); await fsp.unlink(lockPath); }
  }
}

const USAGE = 'Usage: node tools/register_confirmed_phone_review.cjs NEW_AUDIT.json [--captures PATH] [--apply --expected-registry-sha256 SHA256]';
function parseArgs(argv) {
  const [output, ...args] = argv;
  if (!output || output.startsWith('--')) throw Error(USAGE);
  const result = {output: path.resolve(output), apply: false}, seen = new Set();
  for (let i = 0; i < args.length; i++) {
    const option = args[i];
    if (!['--apply', '--captures', '--expected-registry-sha256'].includes(option) || seen.has(option)) throw Error(USAGE);
    seen.add(option);
    if (option === '--apply') result.apply = true;
    else { const value = args[++i]; if (!value || value.startsWith('--')) throw Error(USAGE); result[option === '--captures' ? 'captures' : 'expectedRegistrySha'] = option === '--captures' ? path.resolve(value) : value; }
  }
  if (!inside(ROOT, result.output) || (result.captures && !inside(ROOT, result.captures))) throw Error('CLI audit/capture paths must remain inside this repository');
  return result;
}
if (require.main === module) registerConfirmedReview(parseArgs(process.argv.slice(2))).then(report => console.log(JSON.stringify({status: report.status, assigned: report.assignments.map(item => item.clip), registry_before_sha256: report.registry_before_sha256, preserved_file_count: report.preserved_file_count, preserved_reverified: report.preserved_reverified}))).catch(error => { console.error(error.stack || error.message); process.exitCode = 1; });
module.exports = {registerConfirmedReview, parseArgs, CLIPS, SESSION, SOURCE_SHA, hash};
