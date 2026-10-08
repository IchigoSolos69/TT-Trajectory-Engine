// Hardware smoke test. Keeps only metadata; never saves camera images.
const {spawn}=require('node:child_process');
const fs=require('node:fs');
const assert=require('node:assert/strict');
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function main(){
  const child=spawn('backend_cpp/build/Release/tt_trajectory_engine.exe',['--live']);
  let log='',ws,last;const frames=[];
  child.stdout.on('data',d=>log+=d);child.stderr.on('data',d=>log+=d);
  const wait=async predicate=>{for(let i=0;i<400;i++){if(predicate())return;await delay(50);}throw new Error('Live hardware condition timed out\n'+log);};
  try{
    await wait(()=>log.includes('Listening for WebSocket clients'));
    ws=new WebSocket('ws://localhost:9002');
    ws.onmessage=e=>{last=JSON.parse(e.data);if(last.preview_jpeg)frames.push({frame_id:last.frame_id,mode:last.mode,source_video:last.source_video,width:last.source_width,height:last.source_height,fps:last.fps,preview_fps:last.preview_fps,latency_ms:last.latency_ms,infer_ms:last.infer_ms,jpeg_bytes:last.preview_jpeg.length,people:last.scene_detections.filter(d=>d.class_id===0).length,ball_confidence:last.ball?.confidence??null,table_corners:last.table_corners.length});};
    await wait(()=>last?.mode==='live');assert.equal(last.preview_jpeg,'');
    ws.send('SET_CAMERA:0');ws.send('START_CAMERA');
    await wait(()=>frames.length>=20);
    frames.forEach(f=>{assert.equal(f.mode,'live');assert.equal(f.source_video,'');assert(f.width>0&&f.height>0&&f.jpeg_bytes>1000);});
    ws.send('STOP_CAMERA');await wait(()=>last.camera_active===false && last.preview_jpeg==='');
    const report={passed:true,camera_index:0,camera_released:true,accuracy:'Not evaluated: camera scene has no ground-truth annotations.',frames,log};
    fs.writeFileSync('artifacts/live_camera_runtime.json',JSON.stringify(report,null,2));
    console.log(JSON.stringify({passed:true,frames:frames.length,camera_released:true,mean_capture_fps:frames.reduce((s,f)=>s+f.fps,0)/frames.length},null,2));
  }finally{if(ws?.readyState===WebSocket.OPEN)ws.send('STOP_CAMERA');ws?.close();child.kill();if(child.exitCode===null)await new Promise(r=>child.once('exit',r));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
