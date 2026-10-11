import fs from 'node:fs/promises';
import {createReadStream} from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {FullAbsenceManifest,FullAbsenceFrame,FullAbsenceReview,validateFullAbsence,validIntervalId,VerifiedAbsenceInterval} from './trainingAbsenceIntervals';
import {TrainingRegistry,validCaptureName,validateTrainingRegistry,wholeSourceConflicts} from './trainingSessions';
import {classReviewed,Annotation} from './reviewAnnotations';
import {jpegDimensions} from './reviewGeometry';

export const intervalHash=(raw:string|Buffer)=>createHash('sha256').update(raw).digest('hex');
export async function hashOriginal(file:string){const sha=createHash('sha256');for await(const chunk of createReadStream(file))sha.update(chunk);return sha.digest('hex');}
export async function intervalFile(directory:string,name:string,optional=false){
  try{const file=await fs.realpath(path.join(directory,name));if(path.dirname(file).toLowerCase()!==directory.toLowerCase())throw Error('Interval file escapes its directory');return await fs.readFile(file);}
  catch(error){if(optional&&(error as NodeJS.ErrnoException).code==='ENOENT')return null;throw error;}
}
export async function intervalDirectory(base:string,id:string){
  if(!validIntervalId(id))throw Error('Invalid interval ID');
  const intervals=await fs.realpath(path.join(base,'absence_intervals'));
  if(path.dirname(intervals).toLowerCase()!==base.toLowerCase()||path.basename(intervals).toLowerCase()!=='absence_intervals')throw Error('Interval collection escapes or aliases another directory');
  const directory=await fs.realpath(path.join(intervals,id));
  if(path.dirname(directory).toLowerCase()!==intervals.toLowerCase()||path.basename(directory).toLowerCase()!==id.toLowerCase())throw Error('Interval escapes or aliases another artifact');
  return directory;
}
export async function intervalRegistry(base:string){
  const raw=await intervalFile(base,'training_sessions.json');
  return validateTrainingRegistry(JSON.parse(raw!.toString('utf8')));
}
export async function originalCaptureDirectory(base:string,clip:string){
  if(!validCaptureName(clip))throw Error('Invalid original capture');
  const directory=await fs.realpath(path.join(base,clip));
  if(path.dirname(directory).toLowerCase()!==base.toLowerCase()||path.basename(directory).toLowerCase()!==clip.toLowerCase())throw Error('Original capture identity escapes or aliases another capture');
  return directory;
}
export function checkIntervalAssignment(registry:TrainingRegistry,request:{clip:string;session_id:string;domain:string;role:string;stage:string}){
  if(!validCaptureName(request.clip))throw Error('Invalid interval capture');
  const assignment=registry.captures[request.clip],session=assignment?registry.sessions[assignment.session_id]:null;
  if(!assignment||!session||assignment.session_id!==request.session_id||assignment.domain!==request.domain||session.role!==request.role||session.stage!==request.stage)throw Error('Interval and explicit session/device assignment disagree.');
  if(session.sealed||session.role==='test')throw Error('Sealed session C is reserved for final acceptance. Development preparation and readiness cannot read it.');
  return {assignment,session};
}
export async function intervalSourceConflicts(base:string,registry:TrainingRegistry){
  const metadata:Array<{clip:string;original_sha256?:string}>=[];
  for(const entry of await fs.readdir(base,{withFileTypes:true})){
    if(!entry.isDirectory()||!validCaptureName(entry.name))continue;
    const assignment=registry.captures[entry.name],session=assignment?registry.sessions[assignment.session_id]:undefined;
    if(!assignment||session?.sealed||session?.role==='test')continue;
    const directory=await originalCaptureDirectory(base,entry.name);
    const capture=JSON.parse((await intervalFile(directory,'capture.json'))!.toString('utf8'));metadata.push({clip:entry.name,original_sha256:capture.original_sha256});
  }
  return wholeSourceConflicts(registry,metadata);
}
export async function originalAbsenceConflicts(base:string,registry:TrainingRegistry,source:string,start:number,end:number){
  const conflicts:Array<{clip:string;file:string;source_frame:number}>=[];
  for(const entry of await fs.readdir(base,{withFileTypes:true})){
    if(!entry.isDirectory()||!validCaptureName(entry.name))continue;
    const assignment=registry.captures[entry.name],session=assignment?registry.sessions[assignment.session_id]:null;
    if(session?.sealed||session?.role==='test')continue;
    const directory=await originalCaptureDirectory(base,entry.name);
    const capture=JSON.parse((await intervalFile(directory,'capture.json'))!.toString('utf8'));
    if(capture.original_sha256!==source)continue;
    const rawLabels=await intervalFile(directory,'annotations.json',true);if(!rawLabels)continue;
    const labels=JSON.parse(rawLabels.toString('utf8')) as Record<string,Annotation>;
    const rawIndex=await intervalFile(directory,'frames.jsonl');
    for(const row of rawIndex!.toString('utf8').split('\n').filter(Boolean).map(line=>JSON.parse(line))){
      if(row.source_frame<start||row.source_frame>end)continue;
      const label=labels[row.file];
      if(label&&classReviewed(label,'ball')&&(label.ball?.length||label.geometry?.blur_streak||label.geometry?.visibility!=='absent'))conflicts.push({clip:entry.name,file:row.file,source_frame:row.source_frame});
    }
  }
  return conflicts;
}
function jsonLines(raw:Buffer){const text=raw.toString('utf8'),lines=text.split('\n').slice(0,-1);if(!text.endsWith('\n')||!lines.length||lines.some(line=>!line.trim()))throw Error('Decoder index is incomplete');return lines.map(line=>JSON.parse(line));}
export async function readCompleteInterval(base:string,id:string,registry:TrainingRegistry,requireReview=false){
  const directory=await intervalDirectory(base,id),request=JSON.parse((await intervalFile(directory,'request.json'))!.toString('utf8'));
  checkIntervalAssignment(registry,request); // before any prepared frames or labels
  const jobRaw=await intervalFile(directory,'job.json'),progressRaw=await intervalFile(directory,'progress.json'),cancelRaw=await intervalFile(directory,'cancel.json',true);
  if(JSON.parse(jobRaw!.toString('utf8')).status!=='completed'||JSON.parse(progressRaw!.toString('utf8')).status!=='prepared'||cancelRaw)throw Error('The original decoder job did not finish successfully. Failed, running or canceled data cannot be reviewed or credited.');
  const manifestRaw=(await intervalFile(directory,'manifest.json'))!,manifest=JSON.parse(manifestRaw.toString('utf8')) as FullAbsenceManifest;
  if(manifest.interval_id!==id||manifest.clip!==request.clip||manifest.session_id!==request.session_id||manifest.domain!==request.domain||manifest.role!==request.role||manifest.stage!==request.stage||manifest.start_source_frame!==request.start_source_frame||manifest.end_source_frame!==request.end_source_frame)throw Error('Interval decoder output disagrees with its immutable request');
  checkIntervalAssignment(registry,manifest);
  const indexRaw=(await intervalFile(directory,'frames.jsonl'))!,sourceRaw=(await intervalFile(directory,'source_frames.jsonl'))!;
  const frames=jsonLines(indexRaw) as FullAbsenceFrame[],sourceFrames=jsonLines(sourceRaw);
  const manifestSha=intervalHash(manifestRaw),indexSha=intervalHash(indexRaw),sourceIndexSha=intervalHash(sourceRaw),reviewRaw=await intervalFile(directory,'review.json',true),review=reviewRaw?JSON.parse(reviewRaw.toString('utf8')) as FullAbsenceReview:null;
  validateFullAbsence(manifest,frames,sourceFrames,manifestSha,indexSha,sourceIndexSha,review);
  if(requireReview&&!review)throw Error('Original interval has not been explicitly reviewed as absent');
  const captureDir=await originalCaptureDirectory(base,manifest.clip);
  const captureRaw=(await intervalFile(captureDir,'capture.json'))!,capture=JSON.parse(captureRaw.toString('utf8'));
  if(intervalHash(captureRaw)!==manifest.source_capture_sha256||capture.original_sha256!==manifest.source_sha256||capture.original_path!==manifest.original_path||capture.source!=='recorded_camera'||capture.session_id!==manifest.session_id)throw Error('Original capture identity changed after interval preparation');
  if((await intervalSourceConflicts(base,registry)).has(manifest.source_sha256))throw Error('One original source is assigned to conflicting sessions, roles or devices. All copies are excluded until an audited correction.');
  if(!path.isAbsolute(manifest.resolved_source_path))throw Error('Prepared source location must be an absolute original file');
  if(await hashOriginal(manifest.resolved_source_path)!==manifest.source_sha256)throw Error('Original video bytes changed or are unavailable');
  for(const frame of frames){
    const image=(await intervalFile(directory,frame.file))!,size=jpegDimensions(image);
    if(intervalHash(image)!==frame.image_sha256||size.width!==frame.width||size.height!==frame.height)throw Error('Prepared interval image changed after decoding');
  }
  const conflicts=await originalAbsenceConflicts(base,registry,manifest.source_sha256,manifest.start_source_frame,manifest.end_source_frame);
  if(conflicts.length)throw Error('Existing reviewed ball-positive, hidden or unknown evidence conflicts with this absence interval: '+conflicts.slice(0,8).map(value=>value.source_frame).join(', '));
  return {directory,manifest,frames,review,reviewRaw:reviewRaw?.toString('utf8')??'',manifestSha,indexSha,sourceIndexSha,request};
}

export async function auditedAbsenceIntervals(base:string,registry:TrainingRegistry){
  let entries;try{entries=await fs.readdir(path.join(base,'absence_intervals'),{withFileTypes:true});}catch(error){if((error as NodeJS.ErrnoException).code==='ENOENT')return {intervals:[],issues:[]};throw error;}
  const intervals:VerifiedAbsenceInterval[]=[],issues:string[]=[];
  for(const entry of entries){
    if(!entry.isDirectory()||!validIntervalId(entry.name))continue;
    try{
      const directory=await intervalDirectory(base,entry.name),request=JSON.parse((await intervalFile(directory,'request.json'))!.toString('utf8'));
      const assignment=registry.captures[request.clip],session=assignment?registry.sessions[assignment.session_id]:undefined;
      if(session?.sealed||session?.role==='test')continue; // no manifest/index/review/image read
      if(!await intervalFile(directory,'review.json',true))continue;
      const {manifest,frames}=await readCompleteInterval(base,entry.name,registry,true);
      intervals.push({interval_id:entry.name,clip:manifest.clip,source_sha256:manifest.source_sha256,session_id:manifest.session_id,domain:manifest.domain,role:manifest.role as 'train'|'val',start_source_frame:manifest.start_source_frame,end_source_frame:manifest.end_source_frame,duration_s:manifest.duration_s,frames});
    }catch(error){issues.push(`Absence interval ${entry.name} excluded: ${error instanceof Error?error.message:String(error)}`);}
  }
  return {intervals,issues};
}
