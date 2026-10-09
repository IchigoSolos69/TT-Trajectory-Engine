'use strict';
const fs=require('node:fs');const path=require('node:path');const assert=require('node:assert/strict');const {spawn}=require('node:child_process');const {once}=require('node:events');
async function main(){
  const root=path.resolve(__dirname,'../..');
  const output=path.resolve(process.env.TT_DESKTOP_SMOKE_REPORT||path.join(root,'artifacts','desktop_smoke.json'));
  const executable=process.env.TT_DESKTOP_TEST_EXE||require('electron');
  const fixture=process.env.TT_DESKTOP_SMOKE_VIDEO;
  if(!fixture||!fs.existsSync(fixture))throw Error('Set TT_DESKTOP_SMOKE_VIDEO to an existing short recording');
  const dataRoot=path.join(root,'artifacts',`desktop_test_${Date.now()}`);fs.mkdirSync(dataRoot,{recursive:true});
  const log=fs.createWriteStream(output.replace(/\.json$/,'.log'));
  const childEnv={...process.env,TT_DESKTOP_SMOKE:'1',TT_DESKTOP_SMOKE_REPORT:output,TT_DESKTOP_SMOKE_VIDEO:path.resolve(fixture),TT_DESKTOP_DATA_ROOT:dataRoot};
  if(process.env.TT_DESKTOP_CLEAN_PATH==='1')childEnv.PATH=`${process.env.SystemRoot}\\System32;${process.env.SystemRoot}`;
  const child=spawn(executable,process.env.TT_DESKTOP_TEST_EXE?[]:[path.resolve(__dirname,'..')],{windowsHide:true,stdio:['ignore','pipe','pipe'],env:childEnv});
  child.stdout.pipe(log,{end:false});child.stderr.pipe(log,{end:false});
  const timer=setTimeout(()=>{console.error('Desktop smoke test exceeded five minutes');child.kill();},300000);
  const [code]=await once(child,'exit');clearTimeout(timer);log.end();
  assert.equal(code,0,'Electron smoke process failed; inspect full log');
  const report=JSON.parse(fs.readFileSync(output,'utf8'));assert.equal(report.ok,true,report.error);
  for(const item of [report.live,report.recording])assert.throws(()=>process.kill(item.pid,0),`Backend still running: ${item.pid}`);
  for(const url of [report.live.url,report.recording.url]){await assert.rejects(fetch(url.replace('ws:','http:'),{signal:AbortSignal.timeout(1000)}));}
  await assert.rejects(fetch(report.frontend_origin,{signal:AbortSignal.timeout(1000)}));
  report.checks.push('Closing the native window stops both application-owned services and frees their ports');
  report.clean_runtime_path=process.env.TT_DESKTOP_CLEAN_PATH==='1';
  fs.writeFileSync(output,JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
}
main().catch(error=>{console.error(error.stack);process.exitCode=1;});
