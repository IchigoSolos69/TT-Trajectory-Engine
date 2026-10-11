import {NextRequest,NextResponse} from 'next/server';
import fs from 'node:fs/promises';
import {createWriteStream} from 'node:fs';
import path from 'node:path';
import {randomUUID} from 'node:crypto';
import {spawn} from 'node:child_process';
import {localReviewOrigin,sameOriginReview} from '../../../lib/reviewOrigin';
import {validCaptureName} from '../../../lib/trainingSessions';
import {FullAbsenceReview,validateAbsenceRange,validIntervalId} from '../../../lib/trainingAbsenceIntervals';
import {checkIntervalAssignment,intervalDirectory,intervalFile,intervalHash,intervalRegistry,intervalSourceConflicts,originalAbsenceConflicts,originalCaptureDirectory,readCompleteInterval} from '../../../lib/trainingIntervalStorage';
import {probeIntervalRuntime} from '../../../lib/trainingIntervalRuntime';
import {acquireDecoderLock,releaseDecoderLock} from '../../../lib/trainingDecoderLock';

export const runtime='nodejs';
export const dynamic='force-dynamic';
const root=process.env.TT_TRAINING_ROOT?path.resolve(process.env.TT_TRAINING_ROOT):path.resolve(process.cwd(),'..','training_data','live');
const json=(value:unknown,status=200)=>NextResponse.json(value,{status,headers:{'Cache-Control':'no-store'}});
async function sourceCapture(base:string,clip:string){
  if(!validCaptureName(clip))throw Error('Choose an existing imported capture');
  const registry=await intervalRegistry(base),assignment=registry.captures[clip];if(!assignment)throw Error('Assign the whole session and recording device before preparing an original interval.');
  const session=registry.sessions[assignment.session_id];
  checkIntervalAssignment(registry,{clip,...assignment,role:session.role,stage:session.stage});
  const directory=await originalCaptureDirectory(base,clip);
  const capture=JSON.parse((await intervalFile(directory,'capture.json'))!.toString('utf8'));
  if(capture.source!=='recorded_camera'||typeof capture.original_path!=='string'||!(/^[0-9a-f]{64}$/).test(capture.original_sha256))throw Error('This capture needs its original recorded video and source hash. Sampled webcam images cannot prove continuous footage. Import the original PC-camera recording first.');
  if((await intervalSourceConflicts(base,registry)).has(capture.original_sha256))throw Error('One original source is assigned to conflicting sessions, roles or devices. Disjoint-frame copies cannot become independent data; resolve the provenance before preparation or review.');
  return {registry,assignment,session,capture,directory};
}
async function state(base:string,id:string,registry:Awaited<ReturnType<typeof intervalRegistry>>){
  const directory=await intervalDirectory(base,id),request=JSON.parse((await intervalFile(directory,'request.json'))!.toString('utf8'));
  checkIntervalAssignment(registry,request);
  const jobRaw=await intervalFile(directory,'job.json',true),progressRaw=await intervalFile(directory,'progress.json',true),manifestRaw=await intervalFile(directory,'manifest.json',true),reviewRaw=await intervalFile(directory,'review.json',true);
  const job=jobRaw?JSON.parse(jobRaw.toString('utf8')):{status:'unknown'},progress=progressRaw?JSON.parse(progressRaw.toString('utf8')):null;
  const manifest=manifestRaw?JSON.parse(manifestRaw.toString('utf8')):null;
  const canceled=!!await intervalFile(directory,'cancel.json',true),prepared=manifest?.status==='prepared'&&job.status==='completed'&&progress?.status==='prepared'&&!canceled;
  return {interval_id:id,request,job,progress,prepared,summary:prepared?{interval_frames:manifest.interval_frames,source_decoded_frames:manifest.source_decoded_frames,source_reported_frames:manifest.source_reported_frames,duration_s:manifest.duration_s,coverage_complete:manifest.coverage_complete}:null,reviewed:prepared&&!!reviewRaw};
}
export async function GET(req:NextRequest){
  try{
    localReviewOrigin(req);const base=await fs.realpath(root),id=req.nextUrl.searchParams.get('interval_id'),clip=req.nextUrl.searchParams.get('clip');
    if(!clip)throw Error('Choose a saved capture');const {registry}=await sourceCapture(base,clip);
    if(id){
      const directory=await intervalDirectory(base,id),request=JSON.parse((await intervalFile(directory,'request.json'))!.toString('utf8'));checkIntervalAssignment(registry,request);if(request.clip!==clip)throw Error('Interval does not belong to this capture');
      const file=req.nextUrl.searchParams.get('file');
      if(file){
        if(!/^frame_[0-9]+\.jpg$/.test(file))throw Error('Invalid interval image');
        const index=(await intervalFile(directory,'frames.jsonl'))!.toString('utf8').split('\n').filter(Boolean).map(line=>JSON.parse(line));
        const row=index.find(value=>value.file===file);if(!row)throw Error('Image is not indexed');
        const image=(await intervalFile(directory,file))!;if(intervalHash(image)!==row.image_sha256)throw Error('Prepared image changed');
        return new NextResponse(new Uint8Array(image),{headers:{'Content-Type':'image/jpeg','Cache-Control':'no-store'}});
      }
      if(req.nextUrl.searchParams.get('inspect')==='true'){
        const inspected=await readCompleteInterval(base,id,registry);
        return json({...(await state(base,id,registry)),manifest:inspected.manifest,manifest_sha256:inspected.manifestSha,index_sha256:inspected.indexSha,source_index_sha256:inspected.sourceIndexSha,revision:intervalHash(inspected.reviewRaw),frames:inspected.frames});
      }
      return json(await state(base,id,registry));
    }
    let entries;try{entries=await fs.readdir(path.join(base,'absence_intervals'),{withFileTypes:true});}catch(error){if((error as NodeJS.ErrnoException).code==='ENOENT')return json({intervals:[],preparation_requires_explicit_action:true});throw error;}
    const intervals=[];
    for(const entry of entries){if(!entry.isDirectory()||!validIntervalId(entry.name))continue;const directory=await intervalDirectory(base,entry.name),request=JSON.parse((await intervalFile(directory,'request.json'))!.toString('utf8'));if(request.clip!==clip)continue;intervals.push(await state(base,entry.name,registry));}
    return json({intervals,preparation_requires_explicit_action:true});
  }catch(error){console.error('[training-absence]',error);return json({error:error instanceof Error?error.message:'Cannot read original interval jobs'},400);}
}
export async function POST(req:NextRequest){
  let lock:fs.FileHandle|undefined,lockPath:string|undefined;
  try{
    localReviewOrigin(req);if(!sameOriginReview(req))return json({error:'Same-origin review required'},403);
    const raw=await req.text();if(raw.length>10000)throw Error('Interval request is too large');const body=JSON.parse(raw);
    if(body.action==='probe')return json(await probeIntervalRuntime());
    const base=await fs.realpath(root),{registry,assignment,session,capture}=await sourceCapture(base,body.clip);
    if(body.action==='prepare'){
      if(body.confirm_prepare!==true)throw Error('Confirm preparation of this exact original interval explicitly');
      const range=validateAbsenceRange(body.start_source_frame,body.end_source_frame);
      if(Number.isSafeInteger(capture.source_frames)&&range.end_source_frame>capture.source_frames)throw Error('Interval exceeds imported original source frames');
      const conflicts=await originalAbsenceConflicts(base,registry,capture.original_sha256,range.start_source_frame,range.end_source_frame);
      if(conflicts.length)return json({error:'Existing reviewed ball-positive, hidden or unknown evidence conflicts with absence. Choose an interval that is verified absent; no old annotations were changed.',conflicts},409);
      const runtimeInfo=await probeIntervalRuntime();if(!runtimeInfo.available)return json(runtimeInfo,503);
      const intervals=path.join(base,'absence_intervals');await fs.mkdir(intervals,{recursive:true});const collection=await fs.realpath(intervals);if(path.dirname(collection).toLowerCase()!==base.toLowerCase())throw Error('Interval collection escapes training directory');
      ({lock,lockPath}=await acquireDecoderLock(collection));
      const id=randomUUID(),directory=path.join(collection,id);await fs.mkdir(directory);
      const request={schema_version:1,interval_id:id,clip:body.clip,session_id:assignment.session_id,domain:assignment.domain,role:session.role,stage:session.stage,...range};
      await fs.writeFile(path.join(directory,'request.json'),JSON.stringify(request,null,2)+'\n',{flag:'wx'});
      const job={schema_version:1,status:'running',created_at:new Date().toISOString(),process_id:0,owner_pid:process.pid,error:null as string|null};
      await fs.writeFile(path.join(directory,'job.json'),JSON.stringify(job,null,2)+'\n');
      const log=createWriteStream(path.join(directory,'decode.log'),{flags:'wx'});let loggingError:Error|undefined;
      log.once('error',error=>{
        loggingError=error;
        // This marker belongs only to the worker started for this interval.
        void fs.writeFile(path.join(directory,'cancel.json'),JSON.stringify({schema_version:1,cancel_requested:true})+'\n').catch(value=>console.error('[training-absence log/cancel]',value));
        void fs.writeFile(path.join(directory,'job.json'),JSON.stringify({...job,status:'failed',error:'Decoder log storage failed: '+error.message},null,2)+'\n').catch(value=>console.error('[training-absence log]',value));
      });
      const child=spawn(runtimeInfo.executable,['-B',runtimeInfo.worker,'--captures',base,'--clip',body.clip,'--start-source-frame',String(range.start_source_frame),'--end-source-frame',String(range.end_source_frame),'--interval-id',id,'--owner-pid',String(process.pid)],{windowsHide:true,stdio:['ignore','pipe','pipe'],shell:false});
      child.stdout.pipe(log,{end:false});child.stderr.pipe(log,{end:false});job.process_id=child.pid??0;
      const runningWrite=fs.writeFile(path.join(directory,'job.json'),JSON.stringify(job,null,2)+'\n');
      const lockRaw=JSON.stringify({interval_id:id,process_id:job.process_id,owner_pid:process.pid})+'\n',lockWrite=lock.writeFile(lockRaw);
      const ownedLock=lock,ownedLockPath=lockPath;lock=undefined;lockPath=undefined;
      let finalized=false;
      const finish=async(code:number|null,error?:Error)=>{if(finalized)return;finalized=true;log.end();try{await Promise.all([runningWrite,lockWrite]);const failure=error??loggingError;job.status=code===0&&!failure?'completed':'failed';job.error=failure?.message??(code===0?null:'Decoder exited with code '+code+'. See decode.log; incomplete footage is not credited.');await fs.writeFile(path.join(directory,'job.json'),JSON.stringify({...job,completed_at:new Date().toISOString()},null,2)+'\n');}finally{await releaseDecoderLock(ownedLock,ownedLockPath,lockRaw);}};
      child.once('error',error=>{void finish(null,error).catch(value=>console.error('[training-absence job]',value));});child.once('close',code=>{void finish(code).catch(value=>console.error('[training-absence job]',value));});
      await Promise.all([runningWrite,lockWrite]);
      return json({started:true,interval_id:id,request,job},202);
    }
    if(body.action==='cancel'){
      if(body.confirm_cancel!==true)throw Error('Confirm cancellation explicitly');
      const directory=await intervalDirectory(base,body.interval_id),request=JSON.parse((await intervalFile(directory,'request.json'))!.toString('utf8'));checkIntervalAssignment(registry,request);if(request.clip!==body.clip)throw Error('Interval belongs to another capture');
      if(await intervalFile(directory,'manifest.json',true))throw Error('This decoder has already finished. Prepared data is preserved; it is not reviewed automatically.');
      try{await fs.writeFile(path.join(directory,'cancel.json'),JSON.stringify({schema_version:1,cancel_requested:true})+'\n',{flag:'wx'});}catch(error){if((error as NodeJS.ErrnoException).code!=='EEXIST')throw error;}
      return json({cancel_requested:true,partial_data_credited:false});
    }
    if(body.action==='confirm'){
      if(body.confirm_entire_interval!==true||body.confirm_no_ball!==true||body.no_hidden_or_unsure!==true||body.watched_full_original_interval!==true)throw Error('Explicit review of the entire original interval, verified absence and no hidden/unsure ball are all required');
      const inspected=await readCompleteInterval(base,body.interval_id,registry);if(inspected.manifest.clip!==body.clip)throw Error('Interval belongs to another capture');
      if(inspected.review)return json({error:'This full original interval already has an immutable reviewed record.'},409);
      if(body.manifest_sha256!==inspected.manifestSha||body.index_sha256!==inspected.indexSha||body.source_index_sha256!==inspected.sourceIndexSha||body.revision!==intervalHash(inspected.reviewRaw))return json({error:'The source, decoder audit or review changed. Inspect the original interval again.'},409);
      const m=inspected.manifest,review:FullAbsenceReview={schema_version:1,method:'manual_full_original_interval_absence',reviewed:true,confirm_entire_interval:true,confirm_no_ball:true,no_hidden_or_unsure:true,watched_full_original_interval:true,reviewed_at:new Date().toISOString(),interval_id:m.interval_id,clip:m.clip,source_capture_sha256:m.source_capture_sha256,manifest_sha256:inspected.manifestSha,index_sha256:inspected.indexSha,source_index_sha256:inspected.sourceIndexSha,source_sha256:m.source_sha256,session_id:m.session_id,role:m.role,domain:m.domain,start_source_frame:m.start_source_frame,end_source_frame:m.end_source_frame,duration_s:m.duration_s};
      try{await fs.writeFile(path.join(inspected.directory,'review.json'),JSON.stringify(review,null,2)+'\n',{flag:'wx'});}catch(error){if((error as NodeJS.ErrnoException).code==='EEXIST')return json({error:'Another review saved this interval first. Refresh before continuing.'},409);throw error;}
      return json({saved:true,review,existing_annotations_changed:false});
    }
    throw Error('Choose a supported interval action');
  }catch(error){console.error('[training-absence]',error);return json({error:error instanceof Error?error.message:'Cannot prepare or confirm this original interval'},400);}
  finally{if(lock){await lock.close();if(lockPath)await fs.unlink(lockPath);}}
}
