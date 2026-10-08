const {spawn}=require('node:child_process'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function main(){
 const source=path.resolve('artifacts/replay fixture café.avi');
 const boot=()=>spawn('backend_cpp/build/Release/tt_intelligence_candidate.exe',['--sim','--video',source,'--port','9102','--experimental-intelligence']);
 let child=boot();
 let log='',ws,last;const frames=[];
 child.stdout.on('data',d=>log+=d);child.stderr.on('data',d=>log+=d);
 const wait=async pred=>{for(let i=0;i<2400;i++){if(pred())return;if(child.exitCode!==null)throw Error('Backend exited '+child.exitCode+'\n'+log);await delay(25);}throw Error('Timeout\n'+log+'\nLast packet '+JSON.stringify(last&&{...last,preview_jpeg:undefined}));};
 try{
  await wait(()=>log.includes('Listening for WebSocket clients'));
  ws=new WebSocket('ws://localhost:9102');ws.onmessage=e=>{last=JSON.parse(e.data);frames.push(last);};
  await wait(()=>last?.analysis);
  const id=last.analysis_id;ws.send('PAUSE');await wait(()=>last.paused);
  const startCount=last.analysis.analyzed_frames;
  ws.send('SEEK_FRAME:360');await wait(()=>last.analysis.requested_frame===360||last.frame_id===360);
  const sawWaiting=last.analysis.waiting_for_analysis;
  await wait(()=>last.analysis.status==='COMPLETE'&&last.frame_id===360&&last.paused);
  assert.equal(last.analysis.analyzed_frames,360);assert.equal(last.skipped_frames,0);assert.equal(last.analysis_id,id);
  const end=last;
  ws.send('SEEK_FRAME:120');await wait(()=>last.frame_id===120&&last.paused);
  const selected=last;assert.equal(last.intelligence.trajectory.at(-1).frame_id,120);
  assert(last.strikes.every(s=>s.frame_id<=120),'future strikes leaked into historical replay');
  ws.send('STEP_FORWARD');await wait(()=>last.frame_id===121);ws.send('STEP_BACK');await wait(()=>last.frame_id===120);
  assert.deepEqual(last.intelligence,selected.intelligence);assert.equal(last.preview_jpeg,selected.preview_jpeg);
  for(const speed of [.25,4]){
   ws.send('PLAYBACK_RATE:'+speed);ws.send('SEEK_FRAME:360');await wait(()=>last.frame_id===360&&last.playback_rate===speed);
   assert.deepEqual(last.intelligence,end.intelligence);assert.deepEqual(last.strikes,end.strikes);
  }
  ws.send('LOAD_VIDEO:'+path.resolve('missing-recording.mp4'));await wait(()=>last.video_error);
  assert.equal(last.frame_id,360);assert.equal(last.analysis_id,id);assert.equal(last.preview_jpeg,end.preview_jpeg);
  assert.equal(last.analysis.status,'COMPLETE','Invalid selection must not mark the existing analysis failed');
  ws.send('RESTART');await wait(()=>last.source_id!==end.source_id&&last.frame_id<100);ws.send('PAUSE');await wait(()=>last.paused);
  assert.equal(last.analysis_id,id);assert.equal(last.analysis.analyzed_frames,360);
  const root=path.join('artifacts/analyses',id),rows=fs.readFileSync(path.join(root,'frames.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
  assert.deepEqual(rows.map(r=>r.frame_id),Array.from({length:360},(_,i)=>i+1));
  assert.equal(fs.statSync(path.join(root,'frames.idx')).size,360*8);
  assert(frames.every(f=>f.mode==='simulation'&&!f.camera_active));
  assert(frames.filter(f=>f.intelligence).every(f=>f.intelligence.trajectory.at(-1).frame_id===f.frame_id));
  let writeDenied=false;
  try{const fd=fs.openSync(source,'r+');fs.closeSync(fd);}catch(error){if(!['EBUSY','EACCES','EPERM'].includes(error.code))throw error;writeDenied=true;}
  assert(writeDenied,'Source must remain immutable while its saved analytics are displayed');
  ws.close();child.kill();if(child.exitCode===null)await new Promise(r=>child.once('exit',r));
  const firstLog=log;log='';last=null;child=boot();
  child.stdout.on('data',d=>log+=d);child.stderr.on('data',d=>log+=d);
  await wait(()=>log.includes('Listening for WebSocket clients'));
  ws=new WebSocket('ws://localhost:9102');ws.onmessage=e=>{last=JSON.parse(e.data);};
  await wait(()=>last?.analysis?.status==='COMPLETE');
  assert.equal(last.analysis_id,id,'Relaunch must reuse a compatible completed analysis');
  assert(log.includes(' CACHED'),'Relaunch unexpectedly started analysis again');
  ws.send('PAUSE');ws.send('SEEK_FRAME:120');await wait(()=>last.frame_id===120&&last.paused);
  assert.deepEqual(last.intelligence,selected.intelligence);
  fs.writeFileSync('artifacts/complete_recording_runtime.json',JSON.stringify({passed:true,analysis_id:id,total_frames:360,pause_start_analyzed:startCount,seek_ahead_wait_observed:sawWaiting,unicode_source:true,coverage:'all consecutive source frames',checks:['analysis continues while paused','exact seek and frame stepping','playback-rate independent analytics','historical event state','invalid source preserves frame','restart reuses analysis','source isolation','source write lock','completed cache reused across process restart'],log:firstLog+'\nRELAUNCH\n'+log},null,2));
  console.log('PASS: complete native recording analysis and independent indexed replay');
 }finally{ws?.close();child.kill();if(child.exitCode===null)await new Promise(r=>child.once('exit',r));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
