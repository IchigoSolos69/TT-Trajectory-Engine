'use strict';
const { app, BrowserWindow, Menu, dialog, ipcMain, shell, utilityProcess, nativeImage } = require('electron');
const fs = require('node:fs');
const path = require('node:path');
const { freePort, untilReady, stopChild, EngineSession } = require('./services.cjs');
const { allowedPage } = require('./security.cjs');
const { checkFrontend } = require('./frontend-health.cjs');
const {DraftStore}=require('./draft-store.cjs');

app.setName('TT Trajectory Engine');
app.setAppUserModelId('com.adi.tttrajectory');
if (process.env.TT_DESKTOP_DATA_ROOT) app.setPath('userData', path.resolve(process.env.TT_DESKTOP_DATA_ROOT));
const launcherFile = path.join(__dirname, 'launcher.html');
let window, frontend, engine, origin, closing = false, allowQuit = false, busy = false;
let context = { phase: 'starting', mode: null, backendUrl: null, ballBackend: 'yolo', error: null };
const projectRoot = path.resolve(__dirname, '..');
const resources = app.isPackaged ? path.join(process.resourcesPath, 'runtime') : projectRoot;
function existingWorkspace() {
  let directory=path.dirname(process.execPath);
  for(let depth=0;depth<8;depth++){
    if(fs.existsSync(path.join(directory,'backend_cpp','CMakeLists.txt'))&&fs.existsSync(path.join(directory,'frontend','package.json')))return directory;
    const parent=path.dirname(directory);if(parent===directory)break;directory=parent;
  }
  return null;
}
// Keep this project's existing captures and reviewed annotations when launched
// from its release folder. A moved portable release uses Windows AppData.
const dataRoot = process.env.TT_DESKTOP_DATA_ROOT ? app.getPath('userData') : app.isPackaged ? existingWorkspace() || app.getPath('userData') : projectRoot;
const draftStore=new DraftStore(dataRoot);
const frontendRoot = app.isPackaged ? path.join(resources, 'frontend') : path.join(projectRoot, 'frontend', '.next-desktop', 'standalone');
const executable = app.isPackaged ? path.join(resources, 'backend', 'tt_trajectory_engine.exe') : path.join(projectRoot, 'backend_cpp', 'build', 'Release', 'tt_intelligence_candidate.exe');
const runtimeDirectories = app.isPackaged ? [path.dirname(executable)] : [path.dirname(executable), 'C:/TensorRT/bin', 'C:/opencv/build/x64/vc16/bin', 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/x64'];
function publish(update) {
  context = { ...context, ...update };
  if (window && !window.isDestroyed()) window.webContents.send('desktop:session', { ...context, appVersion: app.getVersion() });
}
function authorize(event) {
  if (!window || event.sender !== window.webContents || event.senderFrame !== window.webContents.mainFrame || !allowedPage(event.senderFrame.url, origin, launcherFile)) throw Error('Untrusted desktop request');
}
async function startSession(mode, video) {
  if (busy || closing) throw Error('A session change is already in progress');
  busy = true; publish({ phase: 'starting', error: null, backendUrl: null });
  try {
    await ensureFrontend();
    const state = await engine.start(mode, video);
    publish({ ...state, phase: 'ready', error: null });
    if (window.webContents.getURL() !== `${origin}/`) await window.loadURL(`${origin}/`);
    return context;
  } catch (error) {
    publish({ ...engine.context(), phase: 'error', error: error.message });
    throw error;
  } finally { busy = false; }
}
async function openRecording() {
  if (busy || closing) return context;
  const selection = await dialog.showOpenDialog(window, {
    title: 'Analyze a game recording', properties: ['openFile'],
    filters: [{ name: 'Game recordings', extensions: ['mov','mp4','avi','mkv','webm','m4v'] }],
  });
  if (selection.canceled || closing) return context;
  return startSession('simulation', selection.filePaths[0]);
}
async function setBallBackend(value) {
  if (busy || closing) throw Error('A session change is already in progress');
  busy=true;publish({phase:'starting',error:null,backendUrl:null});
  try {const state=await engine.setBallBackend(value);publish({...state,phase:state.mode?'ready':'idle',error:null});return context;}
  catch(error){publish({...engine.context(),phase:'error',error:error.message});throw error;}
  finally{busy=false;}
}
async function ensureFrontend() {
  if (frontend) return;
  const server = path.join(frontendRoot, 'server.js');
  if (!fs.existsSync(server)) throw Error('Desktop frontend build missing. Run tools/build_desktop.ps1 first.');
  const port = await freePort(); origin = `http://127.0.0.1:${port}`;
  fs.mkdirSync(path.join(dataRoot, 'logs'), { recursive: true });
  const log = fs.createWriteStream(path.join(dataRoot, 'logs', `dashboard-${Date.now()}.log`));
  frontend = utilityProcess.fork(server, [], {
    cwd: frontendRoot, stdio: 'pipe', serviceName: 'TT local dashboard',
    env: { ...process.env, NODE_ENV: 'production', HOSTNAME: '127.0.0.1', PORT: String(port), NEXT_TELEMETRY_DISABLED: '1', TT_PROJECT_ROOT: dataRoot, TT_TRAINING_ROOT: path.join(dataRoot, 'training_data', 'live') },
  });
  const child = frontend;
  child.stdout.pipe(log, { end: false }); child.stderr.pipe(log, { end: false });
  child.on('exit', code => {
    child.exitCode = code; log.end();
    if (frontend === child) frontend = null;
    if (!child.expectedExit) publish({ phase: 'error', error: `Local dashboard stopped (${code}). See application logs.` });
  });
  try { await untilReady(async () => {
    const health = await checkFrontend(origin);
    if (health.pid !== child.pid || health.root.toLowerCase() !== dataRoot.toLowerCase()) throw Error('Local dashboard identity mismatch');
    return health;
  }, child, 'Local dashboard'); }
  catch (error) { await stopChild(child); throw error; }
}
async function endSession() {
  if (busy) throw Error('A session change is already in progress');
  busy = true;
  try { await engine.stop(); publish({ phase: 'idle', backendUrl: null, mode: null, error: null }); await window.loadFile(launcherFile); }
  finally { busy = false; }
}
async function openTrainingReview() {
  if (busy || closing) throw Error('A workspace change is already in progress');
  busy=true;publish({phase:'starting',error:null});
  try {
    await ensureFrontend();
    await window.loadURL(origin+'/training-review');
    publish({...engine.context(),phase:engine.child?'ready':'idle',error:null});
    return {...context,appVersion:app.getVersion()};
  } catch(error) {publish({...engine.context(),phase:'error',error:error.message});throw error;}
  finally {busy=false;}
}
async function shutdown() {
  if (closing) return;
  closing = true;
  try {
    // Complete a pending startup before stopping its children.
    const deadline = Date.now() + 65000;
    while (busy && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 100));
    await draftStore.pending; await engine?.stop(); await stopChild(frontend);
  } catch (error) { console.error('[desktop] Shutdown:', error); }
  finally { allowQuit = true; app.quit(); }
}
if (!app.requestSingleInstanceLock()) app.quit();
else {
  app.on('second-instance', () => { if (window) { if (window.isMinimized()) window.restore(); window.focus(); } });
  app.on('before-quit', event => { if (!allowQuit) { event.preventDefault(); void shutdown(); } });
  app.whenReady().then(async () => {
    engine = new EngineSession({ executable, ballModel: path.join(resources, 'models', 'yolov11_ttgames.engine'), blurballModel: path.join(resources, 'models', 'blurball_fp32.engine'), sceneModel: path.join(resources, 'models', 'scene.engine'), dataRoot, runtimeDirectories,
      onFailure: error => publish({ phase: 'error', backendUrl: null, error: error.message }) });
    window = new BrowserWindow({ width: 1480, height: 960, minWidth: 960, minHeight: 640,
      title: 'TT Trajectory Engine', backgroundColor: '#020617', show: false,
      icon: nativeImage.createFromPath(path.join(__dirname, 'icon.png')),
      webPreferences: { preload: path.join(__dirname, 'preload.cjs'), nodeIntegration: false, contextIsolation: true, sandbox: true, backgroundThrottling: process.env.TT_DESKTOP_SMOKE!=='1' },
    });
    window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
    window.on('page-title-updated',event=>{event.preventDefault();window.setTitle('TT Trajectory Engine');});
    window.webContents.on('will-navigate', (event, url) => { if (!allowedPage(url, origin, launcherFile)) event.preventDefault(); });
    window.webContents.on('will-attach-webview', event => event.preventDefault());
    window.webContents.session.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
    window.webContents.session.setPermissionCheckHandler(() => false);
    // This application has no remote content or analytics requests.
    window.webContents.session.webRequest.onBeforeRequest((details, callback) => {
      const url = new URL(details.url);
      callback({ cancel: !['file:','data:','blob:','devtools:'].includes(url.protocol) && !((url.hostname === '127.0.0.1') && ([origin, engine.url?.replace('ws:', 'http:')].includes(url.origin.replace('ws:', 'http:')))) });
    });
    ipcMain.handle('desktop:context', event => { authorize(event); return { ...context, appVersion: app.getVersion() }; });
    ipcMain.handle('desktop:recording', event => { authorize(event); return openRecording(); });
    ipcMain.handle('desktop:live', event => { authorize(event); return startSession('live'); });
    ipcMain.handle('desktop:review', event => { authorize(event); return openTrainingReview(); });
    ipcMain.handle('desktop:ball-model', (event,value) => { authorize(event); return setBallBackend(value); });
    ipcMain.handle('desktop:end', event => { authorize(event); return endSession(); });
    ipcMain.handle('desktop:data', async event => { authorize(event); const error = await shell.openPath(dataRoot); if (error) throw Error(error); });
    ipcMain.handle('desktop:drafts-read', event=>{authorize(event);return draftStore.read();});
    ipcMain.handle('desktop:drafts-save', (event,value)=>{authorize(event);return draftStore.save(value);});
    Menu.setApplicationMenu(Menu.buildFromTemplate([
      { label: 'Session', submenu: [
        { label: 'Open recording…', accelerator: 'CmdOrCtrl+O', click: () => void openRecording().catch(error => dialog.showErrorBox('Recording could not open', error.message)) },
        { label: 'Live camera session', click: () => void startSession('live').catch(error => dialog.showErrorBox('Live session could not start', error.message)) },
        { label: 'Training review', click: () => void openTrainingReview().catch(error => dialog.showErrorBox('Training review could not open', error.message)) },
        { label: 'End session', click: () => void endSession().catch(error => dialog.showErrorBox('Session could not stop', error.message)) },
        { type: 'separator' }, { role: 'quit' },
      ] },
      { label: 'View', submenu: [{ role: 'reload' }, { role: 'resetZoom' }, { role: 'zoomIn' }, { role: 'zoomOut' }, { role: 'togglefullscreen' }] },
      { label: 'Help', submenu: [
        { label: 'Open local data and logs', click: () => void shell.openPath(dataRoot) },
        { label: 'About', click: () => void dialog.showMessageBox(window, { title: 'TT Trajectory Engine', message: 'Local table tennis analytics', detail: `Version ${app.getVersion()}\nRecordings and inference stay on this computer.\nAn NVIDIA GPU compatible with the bundled TensorRT engines is required.` }) },
      ] },
    ]));
    window.once('ready-to-show', () => { if(process.env.TT_DESKTOP_SMOKE!=='1')window.show(); });
    window.on('close', event => { if (!allowQuit) { event.preventDefault(); void shutdown(); } });
    await window.loadFile(launcherFile); publish({ phase: 'idle' });
    if (process.env.TT_DESKTOP_SMOKE === '1') await require('./tests/in-app-smoke.cjs').run({ app, window, engine, startSession, endSession, openRecording, setBallBackend, dialog, dataRoot, getOrigin: () => origin, getContext: () => context });
  }).catch(error => { console.error('[desktop]', error); dialog.showErrorBox('Application startup failed', error.stack || error.message); void shutdown(); });
}
