'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { imports } = require('./pe-dependencies.cjs');
require('./icon.cjs');
const root=path.resolve(__dirname,'../..'), desktop=path.join(root,'desktop');
const releaseRoot=path.join(desktop,'dist');
const output=process.env.TT_DESKTOP_OUTPUT_DIR?path.resolve(process.env.TT_DESKTOP_OUTPUT_DIR):path.join(releaseRoot,'TT Trajectory Engine');
if(path.dirname(output)!==releaseRoot)throw Error('Release output must be a direct child of desktop/dist');
const standalone=path.join(root,'frontend','.next-desktop','standalone');
const executable=path.join(root,'backend_cpp','build','Release','tt_intelligence_candidate.exe');
const electronPath=path.join(path.dirname(require.resolve('electron/package.json')),'dist');
const required=[path.join(standalone,'server.js'),executable,path.join(root,'models','yolov11_ttgames.engine'),path.join(root,'models','scene.engine'),path.join(root,'models','blurball_fp32.engine'),path.join(root,'frontend','.next-desktop','BUILD_ID')];
for(const file of required)if(!fs.existsSync(file))throw Error(`Build resource missing: ${file}. Run tools/build_desktop.ps1.`);
const frontendBuild=fs.readFileSync(required[5],'utf8').trim();
if(!frontendBuild)throw Error('Frontend production build is incomplete');
// Refuse to overwrite a running/previous release. Builds use a verified directory.
if(fs.existsSync(output))throw Error(`Release directory already exists: ${output}. Move it aside before packaging a new release.`);
fs.mkdirSync(output,{recursive:true});
fs.cpSync(electronPath,output,{recursive:true});
fs.renameSync(path.join(output,'electron.exe'),path.join(output,'TT Trajectory Engine.exe'));
const appDirectory=path.join(output,'resources','app'), runtime=path.join(output,'resources','runtime');
fs.mkdirSync(appDirectory,{recursive:true});fs.mkdirSync(runtime,{recursive:true});
for(const name of ['package.json','main.cjs','services.cjs','security.cjs','draft-store.cjs','preload.cjs','frontend-health.cjs','launcher.html','launcher.css','launcher.cjs','icon.svg','icon.png','icon.ico'])if(fs.existsSync(path.join(desktop,name)))fs.copyFileSync(path.join(desktop,name),path.join(appDirectory,name));
// Package the native smoke runner too; it is inert unless explicitly enabled.
fs.cpSync(path.join(desktop,'tests'),path.join(appDirectory,'tests'),{recursive:true});
fs.cpSync(standalone,path.join(runtime,'frontend'),{recursive:true});
fs.cpSync(path.join(root,'frontend','.next-desktop','static'),path.join(runtime,'frontend','.next-desktop','static'),{recursive:true});
if(fs.existsSync(path.join(root,'frontend','public')))fs.cpSync(path.join(root,'frontend','public'),path.join(runtime,'frontend','public'),{recursive:true,filter:file=>!(/\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(file))});
fs.mkdirSync(path.join(runtime,'models'),{recursive:true});
for(const name of ['yolov11_ttgames.engine','scene.engine','yolov11_ttgames.manifest.json','scene.manifest.json','blurball_fp32.engine','blurball.manifest.json','BlurBall-LICENSE.txt'])fs.copyFileSync(path.join(root,'models',name),path.join(runtime,'models',name));
const backend=path.join(runtime,'backend');fs.mkdirSync(backend,{recursive:true});
fs.copyFileSync(executable,path.join(backend,'tt_trajectory_engine.exe'));
const searchDirectories=[path.dirname(executable),'C:/TensorRT/bin','C:/opencv/build/x64/vc16/bin','C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/x64',...(process.env.PATH||'').split(path.delimiter)];
const redistRoot='C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Redist/MSVC';
if(fs.existsSync(redistRoot))for(const version of fs.readdirSync(redistRoot))searchDirectories.unshift(path.join(redistRoot,version,'x64','Microsoft.VC143.CRT'));
const files=new Map();
for(const directory of searchDirectories)if(fs.existsSync(directory))for(const name of fs.readdirSync(directory))if(name.toLowerCase().endsWith('.dll')&&!files.has(name.toLowerCase()))files.set(name.toLowerCase(),path.join(directory,name));
const pending=[executable], copied=new Set(), system=[];
// TensorRT/OpenCV dynamically load these plugins; their dependencies are traversed too.
for(const name of ['nvinfer_plugin_11.dll','nvinfer_lean_11.dll','nvinfer_dispatch_11.dll','opencv_videoio_ffmpeg4140_64.dll','cublas64_13.dll','cublasLt64_13.dll']){
  const file=files.get(name.toLowerCase());if(!file)throw Error(`Native runtime not found: ${name}`);pending.push(file);
}
while(pending.length){
  const file=pending.shift(),name=path.basename(file).toLowerCase();
  if(copied.has(name))continue;copied.add(name);
  if(file!==executable)fs.copyFileSync(file,path.join(backend,path.basename(file)));
  for(const dependency of imports(file)){
    const key=dependency.toLowerCase();
    if(/^(api-ms-|ext-ms-)/.test(key)){system.push(dependency);continue;}
    const resolved=files.get(key);
    const systemPath=path.join(process.env.SystemRoot||'C:/Windows','System32',dependency);
    if(resolved&&!resolved.toLowerCase().includes('windows\\system32')&&!resolved.toLowerCase().includes('windows/system32'))pending.push(resolved);
    else if(fs.existsSync(systemPath)){system.push(dependency);}
    else throw Error(`Cannot resolve ${dependency}, required by ${file}`);
  }
}
const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const manifest={schema_version:1,created_at:new Date().toISOString(),platform:'win32-x64',electron:require('electron/package.json').version,frontend_build:frontendBuild,native_runtime:[...copied].sort(),system_dependencies:[...new Set(system)].sort(),source_backend_sha256:hash(executable),models:{ball:hash(required[2]),scene:hash(required[3]),blurball_experimental:hash(required[4])},gpu_requirement:'NVIDIA driver and GPU compatible with the serialized TensorRT engines; built for the current RTX 5050 environment.'};
fs.writeFileSync(path.join(output,'release-manifest.json'),JSON.stringify(manifest,null,2));
fs.writeFileSync(path.join(output,'README.txt'),'Double-click TT Trajectory Engine.exe. Keep this entire folder together.\r\nNo Node.js, Python, browser, or internet connection is required to run.\r\nRequires a compatible NVIDIA GPU/driver. Serialized TensorRT engines are hardware-specific.\r\nChoose a recording or a live session. Camera starts only after explicit authorization.\r\nWhen run inside the source project, existing training_data and analysis folders are retained.\r\nWhen moved outside the project, data and logs use %APPDATA%/TT Trajectory Engine.\r\nUse Help > Open local data and logs to find the active folder.\r\n');
console.log(JSON.stringify({output,native_files:copied.size,manifest},null,2));
