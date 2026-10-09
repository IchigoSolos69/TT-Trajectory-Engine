// Native source-frame replay, not an accuracy evaluation. No camera/service
// is opened and no model or annotation is modified.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {spawn}=require('node:child_process'),assert=require('node:assert/strict');
const source=process.argv[2];
if(!source||!process.argv[3])throw Error('Usage: node test_detection_authority.cjs ORIGINAL_RECORDING OUTPUT_DIRECTORY');
const output=path.resolve(process.argv[3]);
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
async function replay(experimental){
 const directory=path.join(output,experimental?'authority_experimental':'authority_default');
 fs.mkdirSync(directory);fs.writeFileSync(path.join(directory,'request.json'),JSON.stringify({test:'native detection authority',source_sha256:hash(source),experimental}));
 const args=['--sim','--video',path.resolve(source),'--analysis-worker',directory,'--max-frames','1854'];
 if(experimental)args.push('--experimental-intelligence');
 const child=spawn('backend_cpp/build/Release/tt_intelligence_candidate.exe',args);
 let log='';child.stdout.on('data',d=>log+=d);child.stderr.on('data',d=>log+=d);
 const code=await new Promise((resolve,reject)=>{child.on('error',reject);child.on('exit',resolve);});
 fs.writeFileSync(path.join(directory,'runtime.log'),log);
 assert.equal(code,0,log);
 const records=fs.readFileSync(path.join(directory,'frames.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
 assert.equal(records.length,1854);
 assert.equal(JSON.parse(fs.readFileSync(path.join(directory,'completion.json'),'utf8')).status,'PARTIAL');
 let accepted=0,rejected=0,suppressed=0;
 for(let i=0;i<records.length;i++){
  const r=records[i];assert.equal(r.frame_id,i+1);assert.equal(r.scene_frame_id,r.frame_id);
  assert.equal(r.mode,'simulation');assert.equal(r.camera_active,false);assert.equal(r.skipped_frames,0);
  suppressed+=r.suppressed_scene_detections.length;
  if(experimental){
   const s=r.intelligence.trajectory.at(-1);assert.equal(s.frame_id,r.frame_id);
   assert.equal(r.ball!==null,s.accepted,'Raw detector proposal bypassed trajectory authority');
   assert.equal(r.ball_observation.state,s.state);assert.equal(r.ball_observation.reason,s.reason);
   if(r.ball){accepted++;assert(Math.abs(r.ball.x-s.pixel[0]/r.source_width)<=.0001);assert(Math.abs(r.ball.y-s.pixel[1]/r.source_height)<=.0001);}
   if(s.state==='REJECTED'){rejected++;assert.equal(r.ball,null);}
  }else assert.equal(r.ball_observation.state,'DETECTOR_ONLY');
 }
 assert.equal(records[1844].scene_detections.filter(d=>d.class_id===0).length,2,'Reported leg fragment survived');
 assert.equal(records[1844].suppressed_scene_detections.length,1,'Suppressed proposal must remain auditable');
 return {frames:records.length,accepted,rejected,suppressed,reported_frame:records[1847].ball_observation,
   scope:'Native offline source-frame replay; validates data authority and duplicate postprocessing, not ball precision/recall or live latency.'};
}
(async()=>{
 fs.mkdirSync(output,{recursive:true});
 const hashes={source:hash(source),ball:hash('models/yolov11_ttgames.engine'),scene:hash('models/scene.engine')};
 const baseline=await replay(false),experimental=await replay(true);
 assert.deepEqual(hashes,{source:hash(source),ball:hash('models/yolov11_ttgames.engine'),scene:hash('models/scene.engine')});
 fs.writeFileSync(path.join(output,'native_authority.json'),JSON.stringify({passed:true,hashes,baseline,experimental},null,2));
 console.log('PASS: per-frame scene observations, auditable person deduplication, authoritative experimental ball output; source/models unchanged');
})().catch(e=>{console.error(e);process.exitCode=1;});
