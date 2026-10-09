const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const action=process.argv[2],file=process.argv[3];
const ws=new WebSocket('ws://127.0.0.1:9002');
const timeout=setTimeout(()=>{ws.close();throw Error('Native detection runtime timed out');},60000);
let requested=false,seekRequested=false,done=false;
ws.onmessage=e=>{
 if(done)return;
 try{
  const f=JSON.parse(e.data);
  assert.equal(f.mode,'simulation');assert.equal(f.service.application,'TT_TRAJECTORY_ENGINE');
  assert.equal(path.resolve(f.service.root),path.resolve('.'));
  if(action==='snapshot'){
   assert(/^\d+_\d+$/.test(f.analysis_id));
   const request=JSON.parse(fs.readFileSync(path.join('artifacts/analyses',f.analysis_id,'request.json'),'utf8'));
   fs.writeFileSync(file,JSON.stringify({video:request.source_path,frame:f.frame_id,paused:f.paused,rate:f.playback_rate,service:f.service},null,2));
   done=true;ws.send('SHUTDOWN');clearTimeout(timeout);ws.close();console.log('Saved playback state and stopped verified simulation service');
  }else if(action==='restore'){
   const state=JSON.parse(fs.readFileSync(file,'utf8'));
   if(!requested){requested=true;ws.send('LOAD_VIDEO:'+state.video);ws.send('PAUSE');ws.send('PLAYBACK_RATE:'+state.rate);return;}
   if(!seekRequested&&f.source_video===path.basename(state.video)&&f.analysis.analyzed_frames>=1848){
    seekRequested=true;ws.send('SEEK_FRAME:1848');
   }
   if(f.frame_id===1848&&f.source_video===path.basename(state.video)&&f.paused){
    assert.equal(f.scene_frame_id,f.frame_id);assert.equal(f.ball_observation.state,'DETECTOR_ONLY');
    assert.equal(f.scene_detections.filter(d=>d.class_id===0).length,2);
    fs.writeFileSync(path.join(path.dirname(file),'production_runtime.json'),JSON.stringify({...f,preview_jpeg:undefined},null,2));
    done=true;clearTimeout(timeout);ws.close();console.log('PASS: production overlay telemetry at reported frame 1848; current scene observations, two people, explicit detector-only ball candidate');
   }
  }else throw Error('Use snapshot or restore');
 }catch(error){clearTimeout(timeout);ws.close();console.error(error);process.exitCode=1;}
};
// The server closes transport during an authorized SHUTDOWN. Only a completed
// operation may ignore that close race; all earlier connection errors fail.
ws.onerror=e=>{if(done)return;clearTimeout(timeout);console.error(e.error??e);process.exitCode=1;};
