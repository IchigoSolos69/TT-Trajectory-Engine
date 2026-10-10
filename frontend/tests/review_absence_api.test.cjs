'use strict';
// Production route, actual NextRequest, isolated import. No real clips/labels.
const fs=require('node:fs'),fsp=require('node:fs/promises'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const ts=require('../node_modules/typescript'),next=require('../node_modules/next/server');
const load=(file,deps,extra={})=>{const scope={exports:{},require:name=>deps[name],...extra};vm.runInNewContext(ts.transpileModule(fs.readFileSync(file,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,esModuleInterop:true}}).outputText,scope);return scope.exports;};
const lib=path.join(__dirname,'../src/lib'),geometry=load(path.join(lib,'reviewGeometry.ts'),{}),review=load(path.join(lib,'reviewAnnotations.ts'),{'./reviewGeometry':geometry}),absence=load(path.join(lib,'reviewAbsence.ts'),{'./reviewAnnotations':review,'./reviewGeometry':geometry}),origin=load(path.join(lib,'reviewOrigin.ts'),{},{URL});
const sha=text=>crypto.createHash('sha256').update(text).digest('hex');
async function main(){
 const artifacts=path.resolve(__dirname,'../../artifacts'),root=await fsp.mkdtemp(path.join(artifacts,'review-absence-fixture-')),clip='phone_1234567890abcdef_1',dir=path.join(root,clip);await fsp.mkdir(dir);
 const jpg=Buffer.from([0xff,0xd8,0xff,0xc0,0,8,8,4,56,7,128,3,0xff,0xd9]);
 const frames=[1,2,5,6].map((id,i)=>({file:`frame_${i+1}.jpg`,source_frame:id,elapsed_ms:(id-1)*1000/30,burst_id:id<3?1:2}));
 for(const frame of frames)await fsp.writeFile(path.join(dir,frame.file),jpg);
 const capture={source:'recorded_camera',full_source_decode_counted:true,source_frames:6,sampled_frames:4,original_sha256:'a'.repeat(64),session_id:'same_phone_session'};
 const captureRaw=JSON.stringify(capture),indexRaw=frames.map(frame=>JSON.stringify(frame)+'\n').join('');
 await fsp.writeFile(path.join(dir,'capture.json'),captureRaw);await fsp.writeFile(path.join(dir,'frames.jsonl'),indexRaw);
 const scene={reviewed:true,reviewed_classes:{ball:false,person:true,table:true},ball:[],person:[[.1,.1,.2,.4]],table:[[.2,.5,.5,.2]],geometry:{...geometry.emptyGeometry(),landmarks:{net_left:[.2,.5]},landmarks_reviewed:true,events:['rally_end'],events_reviewed:true}};
 let raw=JSON.stringify({'frame_1.jpg':scene,'frame_2.jpg':{reviewed:false,ball:[],person:[],table:[]},'frame_3.jpg':{...scene,ball:[[.4,.4,.02,.02]],geometry:{...scene.geometry,visibility:'visible'}},'frame_4.jpg':{...scene,geometry:{...scene.geometry,visibility:'occluded'}}});
 await fsp.writeFile(path.join(dir,'annotations.json'),raw);
 const errors=[],route=load(path.join(__dirname,'../src/app/api/training-review/route.ts'),{
  'next/server':next,'node:fs/promises':fsp,'node:path':path,'node:crypto':crypto,
  '../../../lib/reviewGeometry':geometry,'../../../lib/reviewOrigin':origin,'../../../lib/reviewAbsence':absence,
 },{process:{env:{TT_TRAINING_ROOT:root},cwd:()=>path.resolve(__dirname,'../..')},console:{error(...args){errors.push(args.map(value=>value?.stack??String(value)).join(' '));}},Error});
 let revision=sha(raw);
 const request=async(action,extra={},originHeader='http://127.0.0.1:3100')=>{
  const req=new next.NextRequest('http://127.0.0.1:3100/api/training-review',{method:'POST',headers:{host:'127.0.0.1:3100',origin:originHeader,'content-type':'application/json'},body:JSON.stringify({clip,revision,start_source_frame:1,end_source_frame:2,action,...extra})});
  const response=await route.POST(req);return {status:response.status,body:await response.json()};
 };
 const checkStatus=(result,status)=>assert.equal(result.status,status,JSON.stringify(result.body)+'\n'+errors.join('\n'));
 try{
  const preview=await request('preview_ball_absence_interval');checkStatus(preview,200);assert.equal(preview.body.affected_frames,2);assert.deepEqual(preview.body.frames.map(f=>f.source_frame),[1,2]);assert.equal(preview.body.conflicts.length,0);
  assert.equal(await fsp.readFile(path.join(dir,'annotations.json'),'utf8'),raw,'Preview must not change labels');assert(!fs.existsSync(path.join(dir,'annotation_history')));
  checkStatus(await request('confirm_ball_absence_interval',{preview_token:preview.body.preview_token}),400);
  checkStatus(await request('confirm_ball_absence_interval',{preview_token:'wrong',confirm_absence:true}),409);
  checkStatus(await request('preview_ball_absence_interval',{},'http://localhost:3100'),403);
  checkStatus(await request('preview_ball_absence_interval',{revision:'stale'}),409);
  checkStatus(await request('preview_ball_absence_interval',{start_source_frame:2,end_source_frame:1}),400);
  checkStatus(await request('preview_ball_absence_interval',{end_source_frame:7}),400);
  checkStatus(await request('preview_ball_absence_interval',{start_source_frame:3,end_source_frame:4}),400);
  await fsp.writeFile(path.join(dir,'frame_1.jpg'),Buffer.concat([jpg,Buffer.from([1])]));
  checkStatus(await request('confirm_ball_absence_interval',{preview_token:preview.body.preview_token,confirm_absence:true}),409);await fsp.writeFile(path.join(dir,'frame_1.jpg'),jpg);
  await fsp.writeFile(path.join(dir,'capture.json'),JSON.stringify({...capture,session_id:'different'}));
  checkStatus(await request('confirm_ball_absence_interval',{preview_token:preview.body.preview_token,confirm_absence:true}),409);await fsp.writeFile(path.join(dir,'capture.json'),captureRaw);
  await fsp.writeFile(path.join(dir,'frames.jsonl'),indexRaw.slice(0,-1));checkStatus(await request('preview_ball_absence_interval'),400);await fsp.writeFile(path.join(dir,'frames.jsonl'),indexRaw);
  await fsp.writeFile(path.join(dir,'capture.json'),JSON.stringify({...capture,sampled_frames:5}));checkStatus(await request('preview_ball_absence_interval'),400);await fsp.writeFile(path.join(dir,'capture.json'),captureRaw);
  const conflicting=await request('preview_ball_absence_interval',{start_source_frame:5,end_source_frame:6});checkStatus(conflicting,200);assert.deepEqual(conflicting.body.conflicts.map(f=>f.source_frame),[5,6]);
  checkStatus(await request('confirm_ball_absence_interval',{start_source_frame:5,end_source_frame:6,preview_token:conflicting.body.preview_token,confirm_absence:true}),409);
  assert.equal(await fsp.readFile(path.join(dir,'annotations.json'),'utf8'),raw,'Invalid/conflicting requests leave all original labels intact');
  const saved=await request('confirm_ball_absence_interval',{preview_token:preview.body.preview_token,confirm_absence:true});checkStatus(saved,200);
  for(const name of ['frame_1.jpg','frame_2.jpg']){const label=saved.body.annotations[name];assert.equal(label.geometry.visibility,'absent');assert.equal(label.geometry.blur_streak,null);assert.deepEqual(label.ball,[]);assert.equal(label.reviewed_classes.ball,true);assert.equal(label.ball_absence_review.image_sha256,sha(jpg));assert.equal(label.ball_absence_review.original_sha256,capture.original_sha256);assert.equal(label.ball_absence_review.previous_annotation_revision,revision);}
  const preserved=saved.body.annotations['frame_1.jpg'];assert.deepEqual(preserved.person,scene.person);assert.deepEqual(preserved.table,scene.table);assert.deepEqual(preserved.geometry.landmarks,scene.geometry.landmarks);assert.deepEqual(preserved.geometry.events,scene.geometry.events);assert.equal(preserved.reviewed_classes.person,true);assert.equal(preserved.reviewed_classes.table,true);
  const fresh=saved.body.annotations['frame_2.jpg'];assert.equal(fresh.reviewed_classes.person,false);assert.equal(fresh.reviewed_classes.table,false);
  assert.deepEqual(saved.body.annotations['frame_3.jpg'],JSON.parse(raw)['frame_3.jpg']);assert.deepEqual(saved.body.annotations['frame_4.jpg'],JSON.parse(raw)['frame_4.jpg']);
  assert.equal(await fsp.readFile(path.join(dir,'annotation_history',revision+'.json'),'utf8'),raw);assert(!fs.existsSync(path.join(dir,'.review.lock')));
  checkStatus(await request('confirm_ball_absence_interval',{preview_token:preview.body.preview_token,confirm_absence:true}),409);
  const read=await route.GET(new next.NextRequest('http://127.0.0.1:3100/api/training-review?clip='+clip,{headers:{host:'127.0.0.1:3100'}}));assert.equal((await read.json()).annotations['frame_1.jpg'].geometry.visibility,'absent');
  await fsp.writeFile(path.join(artifacts,'review_absence_api_checks.json'),JSON.stringify({passed:true,scope:'Production API with actual NextRequest and isolated fixture; no real clip import or browser rendering',checks:['read-only exact sampled-frame preview','explicit confirmation','source/image/index/session-bound token','stale revision and token rejection','incomplete index/count rejection','visible/occluded evidence conflicts','scene/landmarks/events preservation','new scene classes remain unknown','immutable history','reload and lock cleanup','secure origin']},null,2)+'\n');
  console.log('PASS: bulk ball absence production API, source-bound preview, conflict/CSRF/revision protection, scene preservation and immutable history.');
 }finally{if(path.dirname(root)!==artifacts||!path.basename(root).startsWith('review-absence-fixture-'))throw Error('Fixture cleanup path unsafe');await fsp.rm(root,{recursive:true,force:true});}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
