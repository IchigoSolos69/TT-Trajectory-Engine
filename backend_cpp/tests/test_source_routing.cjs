// Integration against the compiled production executable and real WebSocket payloads.
const { spawn, spawnSync } = require('node:child_process');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const exe = 'backend_cpp/build/Release/tt_trajectory_engine.exe';
const report = [];

for (const args of [['--sim', '--live'], ['--live', '--video', 'python_tools/dataset/raw_videos/game_1.mp4'],
  ['--sim', '--video', 'missing_dataset.mp4'], ['--sim', 'missing_model.engine']]) {
  const run = spawnSync(exe, args, { encoding: 'utf8', timeout: 20000 });
  assert.equal(run.status, 1, run.stdout + run.stderr);
  assert.match(run.stderr, /FATAL/);
  report.push({ args, exit: run.status, error: run.stderr.trim() });
}

async function capture(mode) {
  const args = [mode,'--port','9103'];
  if (mode === '--sim') args.push('--video','artifacts/replay fixture café.avi');
  const child = spawn(exe, args, { stdio: ['ignore', 'pipe', 'pipe'] });
  let log = '', ws;
  const frames = [];
  try {
    await new Promise((resolve, reject) => {
      const deadline = setTimeout(() => reject(new Error('Timed out waiting for telemetry: ' + log)), 30000);
      child.on('error', reject);
      child.on('exit', code => { if (frames.length < 3) reject(new Error('Premature exit ' + code + log)); });
      child.stderr.on('data', data => { log += data; });
      child.stdout.on('data', data => {
        log += data;
        if (ws || !log.includes('Listening for WebSocket clients')) return;
        ws = new WebSocket('ws://localhost:9103');
        ws.onopen = () => {
          if (mode === '--sim') ws.send('START_CAMERA');
          ws.send('START_TRAINING_RECORDING');
        };
        ws.onmessage = event => {
          try {
            const frame = JSON.parse(event.data);
            if(mode==='--sim' && frame.frame_id===0){
              assert.equal(frame.analysis.waiting_for_analysis,true);
              assert.equal(frame.camera_active,false);
              assert.equal(frame.preview_jpeg,'');
              return; // Analysis-loading packets are not analyzed source frames.
            }
            // Keep metadata as evidence without duplicating footage in reports.
            frames.push({ ...frame, preview_jpeg: frame.preview_jpeg ? `[JPEG ${frame.preview_jpeg.length} base64 bytes]` : '' });
            assert.equal(frame.mode, mode === '--sim' ? 'simulation' : 'live');
            assert.deepEqual(frame.players, []);
            assert.equal(frame.training_recording.active, false, 'simulation/unauthorized camera must not record');
            assert.equal(frame.training_recording.clip, '');
            if (mode === '--live') {
              assert.equal(frame.frame_id, 0);
              assert.equal(frame.fps, 0);
              assert.equal(frame.ball, null);
              assert.equal(frame.source_video, '');
              assert.equal(frame.preview_jpeg, '');
            } else {
              assert.equal(frame.source_video, 'replay fixture café.avi');
              assert.equal(frame.fps, 120);
              assert.ok(frame.frame_id > 0 && frame.frame_id <= 360);
              assert.ok(frame.latency_ms > 0 && frame.infer_ms > 0);
              assert.ok(frame.preview_jpeg.length > 1000);
              assert.ok(Array.isArray(frame.scene_detections));
            }
            if (frames.length === 3) { clearTimeout(deadline); resolve(); }
          } catch (error) { clearTimeout(deadline); reject(error); }
        };
        ws.onerror = error => { clearTimeout(deadline); reject(error); };
      });
    });
    if (mode === '--live') assert.ok(!log.includes('Dataset source:') && !log.includes('Booting webcam'));
    report.push({ args, frames, log });
  } finally {
    if (ws) ws.close();
    child.kill();
    if(child.exitCode===null) await new Promise(resolve => child.once('exit', resolve));
  }
}

(async () => {
  await capture('--sim');
  await capture('--live');
  fs.mkdirSync('artifacts/detector_improvement', { recursive: true });
  fs.writeFileSync('artifacts/detector_improvement/source_routing.json', JSON.stringify(report, null, 2));
  console.log('PASS: 4 invalid-source cases; real dataset telemetry; live standby without camera authorization.');
})().catch(error => { console.error(error); process.exitCode = 1; });
