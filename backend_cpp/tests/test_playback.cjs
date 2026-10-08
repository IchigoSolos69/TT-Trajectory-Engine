const {spawn}=require('node:child_process');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function main(){
  const child=spawn('backend_cpp/build/Release/tt_trajectory_engine.exe',['--sim','--video','artifacts/replay fixture café.avi','--port','9105']);
  let log='',ws,last,decodeErrors=0;const samples=[];
  child.stdout.on('data',d=>log+=d);child.stderr.on('data',d=>log+=d);
  const wait=async predicate=>{for(let i=0;i<300;i++){if(predicate())return;await delay(50);}throw new Error('Runtime condition timed out\n'+log);};
  try{
    await wait(()=>log.includes('Listening for WebSocket clients'));
    ws=new WebSocket('ws://localhost:9105');
    ws.onmessage=e=>{try{last=JSON.parse(e.data);}catch{decodeErrors++;}};
    await wait(()=>last?.preview_jpeg);
    await wait(()=>last.analysis?.status==='COMPLETE');
    const rates=[];
    for(const rate of [.25,1,2,4]){
      ws.send('PAUSE');await wait(()=>last.paused);
      ws.send(`PLAYBACK_RATE:${rate}`);await wait(()=>last.playback_rate===rate);
      const epoch=last.source_id;
      ws.send('RESTART');await wait(()=>last.source_id!==epoch&&!last.paused&&last.frame_id<120);
      await delay(150);
      const start={time:performance.now(),source:last.source_time,frame:last.frame_id};
      await delay(400);
      const measured=(last.source_time-start.source)/((performance.now()-start.time)/1000);
      assert(Math.abs(measured-rate)/rate<.3,`Playback ${rate}x achieved ${measured}x`);
      assert.equal(last.analysis.analyzed_frames,360);assert.equal(last.skipped_frames,0);
      rates.push({requested:rate,achieved:measured,analyzed_fps:last.processing_fps,preview_fps:last.preview_fps,skipped_frames:last.skipped_frames});
    }
    ws.send('PAUSE');await wait(()=>last.paused);
    const frozen=last;
    await delay(400);
    assert.equal(last.frame_id,frozen.frame_id);assert.equal(last.preview_jpeg,frozen.preview_jpeg);
    assert.deepEqual(last.ball,frozen.ball,'pause must retain matching detections');
    fs.writeFileSync('artifacts/preview_alignment.jpg',Buffer.from(frozen.preview_jpeg,'base64'));
    fs.writeFileSync('artifacts/preview_alignment.json',JSON.stringify({...frozen,preview_jpeg:undefined},null,2));
    assert.equal(decodeErrors,0,'partial socket writes must not corrupt JSON');
    samples.push({rates,pause_frame:last.frame_id,decode_errors:decodeErrors});
    fs.writeFileSync('artifacts/playback_runtime.json',JSON.stringify({passed:true,samples,log},null,2));
    console.log(JSON.stringify({passed:true,rates,pause_frame:last.frame_id},null,2));
  }finally{
    ws?.close();child.kill();
    if(child.exitCode===null)await new Promise(r=>child.once('exit',r));
  }
}
main().catch(e=>{console.error(e);process.exitCode=1;});
