'use strict';
const net = require('node:net');
const fs = require('node:fs');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { once } = require('node:events');
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const ended = child => child.exited || (child.exitCode !== null && child.exitCode !== undefined) || (child.signalCode !== null && child.signalCode !== undefined);

async function freePort() {
  const server = net.createServer();
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const port = server.address().port;
  await new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  return port;
}

async function untilReady(check, child, description, timeout = 60000) {
  const start = Date.now(); let lastError;
  while (Date.now() - start < timeout) {
    if (ended(child)) throw Error(`${description} exited (${child.exitCode ?? child.signalCode}). See the application logs.`);
    if (child.spawnError) throw child.spawnError;
    try { return await check(); } catch (error) { lastError = error; }
    await delay(200);
  }
  throw Error(`${description} did not become ready: ${lastError?.message || 'timeout'}`);
}

function enginePacket(url, command, expectedPid, timeout = 5000) {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(url); let finished = false, received = false, firstPacket;
    const end = (error, packet) => {
      if (finished) return; finished = true; clearTimeout(timer); socket.close();
      error ? reject(error) : resolve(packet);
    };
    const timer = setTimeout(() => end(Error('Engine connection timed out')), timeout);
    socket.onerror = () => { if(command==='SHUTDOWN'&&firstPacket)end(null,firstPacket);else end(Error('Engine connection failed')); };
    socket.onclose = () => { if(!finished){if(command==='SHUTDOWN'&&firstPacket)end(null,firstPacket);else end(Error('Engine closed before reporting its identity'));} };
    socket.onmessage = event => {
      if (received) return; received = true;
      try {
        const packet = JSON.parse(event.data);
        if (packet.service?.application !== 'TT_TRAJECTORY_ENGINE' || packet.service.pid !== expectedPid) throw Error('Engine ownership mismatch');
        if (command) socket.send(command);
        firstPacket=packet;
        // Let queued shutdown/load bytes flush before closing the socket.
        setTimeout(() => end(null, packet), command ? 100 : 0);
      } catch (error) { end(error); }
    };
  });
}

async function stopChild(child, shutdown) {
  if (!child || ended(child)) return;
  if (child.spawnError && !child.pid) return;
  child.expectedExit = true;
  if (shutdown) await shutdown();
  if (ended(child)) return;
  const exited = once(child, 'exit');
  if (await Promise.race([exited.then(() => true), delay(shutdown ? 8000 : 100).then(() => false)])) return;
  // This handle refers only to a process spawned by this instance. No port-owner killing.
  child.kill();
  if (!await Promise.race([exited.then(() => true), delay(5000).then(() => false)])) throw Error('Application-owned service did not stop');
}

class EngineSession {
  constructor({ executable, ballModel, blurballModel, sceneModel, dataRoot, runtimeDirectories = [], onFailure = () => {} }) {
    Object.assign(this, { executable, ballModel, blurballModel, sceneModel, dataRoot, runtimeDirectories, onFailure });
    this.child = null; this.url = null; this.mode = null; this.video = null; this.ballBackend = 'yolo';
  }
  async start(mode, video) {
    if (!['simulation', 'live'].includes(mode)) throw Error('Unknown source mode');
    if (mode === 'live' && video) throw Error('Live mode cannot load a recording');
    if (mode === 'simulation' && (!video || !fs.statSync(video).isFile() || !/\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(video))) throw Error('Select an existing supported recording');
    const model = this.ballBackend === 'blurball' ? this.blurballModel : this.ballModel;
    for (const required of [this.executable, model, this.sceneModel]) if (!required || !fs.existsSync(required)) throw Error(`Required application resource missing: ${required}`);
    if (mode === 'simulation' && this.mode === mode && this.child && !ended(this.child)) {
      await enginePacket(this.url, `LOAD_VIDEO:${video}`, this.child.pid); this.video=video; return this.context();
    }
    await this.stop();
    const port = await freePort();
    fs.mkdirSync(path.join(this.dataRoot, 'logs'), { recursive: true });
    const log = fs.createWriteStream(path.join(this.dataRoot, 'logs', `engine-${Date.now()}.log`));
    const args = [mode === 'live' ? '--live' : '--sim', '--port', String(port), '--data-root', this.dataRoot, '--ball-backend', this.ballBackend, model, '--scene-engine', this.sceneModel];
    if (video) args.push('--video', video);
    const child = spawn(this.executable, args, {
      cwd: this.dataRoot, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, PATH: [...this.runtimeDirectories, process.env.PATH || ''].join(path.delimiter), CUDA_VISIBLE_DEVICES: '0' },
    });
    this.child = child; this.url = `ws://127.0.0.1:${port}`; this.mode = mode; this.video=video || null;
    child.on('error', error => { child.spawnError = error; this.onFailure(error); });
    child.stdout.pipe(log, { end: false }); child.stderr.pipe(log, { end: false });
    child.on('exit', (code, signal) => { log.end(); if (!child.expectedExit) this.onFailure(Error(`Engine stopped (${code ?? signal}). See the application logs.`)); });
    try {
      await untilReady(async () => {
        const packet = await enginePacket(this.url, null, child.pid, 1000);
        if (packet.mode !== mode || packet.service.root.toLowerCase() !== this.dataRoot.toLowerCase()) throw Error('Engine source/data-directory mismatch');
        return packet;
      }, child, 'C++ engine');
      return this.context();
    } catch (error) { await this.stop(); throw error; }
  }
  context() { return { backendUrl: this.url, mode: this.mode, ballBackend: this.ballBackend }; }
  async setBallBackend(value) {
    if (!['yolo','blurball'].includes(value)) throw Error('Unknown ball model');
    const model=value==='blurball'?this.blurballModel:this.ballModel;
    if (!model || !fs.existsSync(model)) throw Error('Selected ball model is not installed');
    if (value===this.ballBackend) return this.context();
    const mode=this.mode, video=this.video;
    await this.stop(); this.ballBackend=value;
    return mode ? this.start(mode, video) : this.context();
  }
  async stop() {
    const child = this.child, url = this.url;
    if (!child) return;
    await stopChild(child, async () => {
      try { await enginePacket(url, 'SHUTDOWN', child.pid, 1500); }
      catch (error) { if (child.exitCode === null) console.error('[desktop] Graceful engine shutdown unavailable:', error.message); }
    });
    this.child = null; this.url = null; this.mode = null; this.video=null;
  }
}
module.exports = { freePort, untilReady, enginePacket, stopChild, EngineSession };
