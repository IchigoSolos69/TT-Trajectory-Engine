const {spawn}=require('node:child_process');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function session(mode, run){
  const child=spawn('backend_cpp/build/Release/tt_trajectory_engine.exe',[mode]);
  let log='',last,ws;const frames=[];
  child.stdout.on('data',d=>log+=d);child.stderr.on('data',d=>log+=d);
  const wait=async pred=>{for(let i=0;i<600;i++){if(pred())return;await delay(50);}throw Error('Timed out\n'+log);};
  try{
    await wait(()=>log.includes('Listening for WebSocket clients'));
    ws=new WebSocket('ws://localhost:9002');ws.onmessage=e=>{last=JSON.parse(e.data);frames.push({...last,preview_jpeg:undefined});};
    await wait(()=>last);
    await run({wait,send:cmd=>ws.send(cmd),get:()=>last,frames});
    return {mode,frames:frames.length,sample:frames.at(-1),log};
  }finally{ws?.close();child.kill();if(child.exitCode===null)await new Promise(r=>child.once('exit',r));}
}
async function main(){
 const sim=await session('--sim',async({wait,send,get,frames})=>{
   await wait(()=>get()?.preview_jpeg);
   send('PLAYBACK_RATE:4');
   await wait(()=>get()?.strikes.length>0);
   send('PAUSE');await wait(()=>get().paused);
   const before=get();
   assert.equal(before.spin.rpm,null);assert.equal(before.height_observed,false);assert.equal(before.ball?.z??0,0);
   assert(before.strikes.every(s=>s.speed_kmh>0 && s.frame_id<=before.frame_id));
   await delay(200);assert.deepEqual(get().strikes,before.strikes,'strike history persists on paused frames');
   send('LOAD_VIDEO:'+path.resolve('missing-recording.mp4'));
   await wait(()=>get().video_error);
   assert.equal(get().source_id,before.source_id);assert.equal(get().frame_id,before.frame_id);
   send('LOAD_VIDEO:'+path.resolve('python_tools/dataset/raw_videos/test_2.mp4'));
   await wait(()=>get().source_video==='test_2.mp4' && get().preview_jpeg && get().source_id!==before.source_id);
   assert.equal(get().strikes.length,0,'new source clears old strikes');assert.equal(get().video_error,'');
   send('PAUSE');await wait(()=>get().paused);
   const changed=get();await delay(200);assert.equal(get().frame_id,changed.frame_id);assert.equal(get().preview_jpeg,changed.preview_jpeg);
   fs.writeFileSync('artifacts/spatial_runtime_frame.json',JSON.stringify({...before,preview_jpeg:undefined},null,2));
   fs.writeFileSync('artifacts/spatial_runtime_frame.jpg',Buffer.from(before.preview_jpeg,'base64'));
   assert(frames.every(f=>f.mode==='simulation' && !f.camera_active));
 });
 const live=await session('--live',async({wait,send,get})=>{
   send('LOAD_VIDEO:'+path.resolve('python_tools/dataset/raw_videos/test_2.mp4'));
   await wait(()=>get().video_error);
   assert.equal(get().mode,'live');assert.equal(get().camera_active,false);assert.equal(get().preview_jpeg,'');
   assert.equal(get().source_video,'');assert.equal(get().ball,null);
 });
 fs.writeFileSync('artifacts/recording_selection_runtime.json',JSON.stringify({passed:true,sim,live},null,2));
 console.log('PASS: persistent strike speeds, unresolved spin, source switch/reset, invalid file preservation, paused frame alignment, live source isolation.');
}
main().catch(e=>{console.error(e);process.exitCode=1;});
