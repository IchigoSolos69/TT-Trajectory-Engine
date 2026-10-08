const {spawn}=require('node:child_process'),assert=require('node:assert/strict'),fs=require('node:fs');
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function main(){
 const child=spawn('backend_cpp/build/Release/tt_intelligence_candidate.exe',['--sim','--experimental-intelligence','--port','9102']);
 let log='',ws,last;const samples=[];
 child.stdout.on('data',d=>log+=d);child.stderr.on('data',d=>log+=d);
 const wait=async fn=>{for(let i=0;i<500;i++){if(fn())return;await delay(50);}throw Error('Timed out\n'+log);};
 try{
  await wait(()=>log.includes('Listening for WebSocket clients'));
  ws=new WebSocket('ws://localhost:9102');ws.onmessage=e=>{last=JSON.parse(e.data);samples.push(last);};
  await wait(()=>last?.intelligence?.calibration.state==='LOCKED');
  await wait(()=>last.intelligence.trajectory.some(s=>s.state==='PREDICTED'||s.state==='REJECTED'));
  assert(samples.every(f=>f.intelligence.trajectory.at(-1).frame_id===f.frame_id),'latest trajectory must match image frame');
  assert(samples.every(f=>f.intelligence.trajectory.length<=180),'history must be bounded');
  assert(samples.every(f=>f.intelligence.provenance.height==='UNOBSERVED'));
  fs.writeFileSync('artifacts/intelligence_ui_sample.json',JSON.stringify(last,null,2));
  ws.send('SEEK_FRAME:120');await wait(()=>last.paused&&last.frame_id===120);
  const frozen=last;await delay(250);assert.equal(last.preview_jpeg,frozen.preview_jpeg);
  assert.equal(last.intelligence.trajectory.at(-1).frame_id,120);
  ws.send('SEEK_FRAME:999999999');await wait(()=>last.video_error);assert.equal(last.frame_id,120);
  assert.equal(last.mode,'simulation');assert.equal(last.camera_active,false);
  fs.writeFileSync('artifacts/intelligence_runtime_frame.json',JSON.stringify(frozen,null,2));
  fs.writeFileSync('artifacts/intelligence_runtime.json',JSON.stringify({passed:true,frames:samples.length,states:[...new Set(samples.flatMap(f=>f.intelligence.trajectory.map(s=>s.state)))],seek_frame:120,log},null,2));
  console.log('PASS: native inference, calibration lock, trajectory states/provenance, bounded history, exact-frame seek/pause, invalid seek preservation.');
 }finally{ws?.close();child.kill();if(child.exitCode===null)await new Promise(r=>child.once('exit',r));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
