import {NextRequest,NextResponse} from 'next/server';
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash,randomUUID} from 'node:crypto';
import {localReviewOrigin,sameOriginReview} from '../../../lib/reviewOrigin';
import {assignTrainingCapture,ContinuityAudit,emptyTrainingRegistry,ReadinessCapture,trainingReadiness,validateTrainingRegistry,validCaptureName,wholeSourceConflicts} from '../../../lib/trainingSessions';
import {auditedAbsenceIntervals} from '../../../lib/trainingIntervalStorage';

export const runtime='nodejs';
export const dynamic='force-dynamic';
const root=process.env.TT_TRAINING_ROOT?path.resolve(process.env.TT_TRAINING_ROOT):path.resolve(process.cwd(),'..','training_data','live');
const registryFile='training_sessions.json';
const hash=(raw:string|Buffer)=>createHash('sha256').update(raw).digest('hex');
async function readRegistry(base:string){
  let raw='';
  try{
    const file=await fs.realpath(path.join(base,registryFile));
    if(path.dirname(file).toLowerCase()!==base.toLowerCase())throw Error('Registry escapes the training directory');
    raw=await fs.readFile(file,'utf8');
  }catch(error){if((error as NodeJS.ErrnoException).code!=='ENOENT')throw error;}
  return {raw,revision:hash(raw),registry:raw?validateTrainingRegistry(JSON.parse(raw)):emptyTrainingRegistry()};
}
async function clipDirectory(base:string,clip:string){
  if(!validCaptureName(clip))throw Error('Invalid capture');
  const directory=await fs.realpath(path.join(base,clip));
  if(path.dirname(directory).toLowerCase()!==base.toLowerCase()||path.basename(directory).toLowerCase()!==clip.toLowerCase())throw Error('Capture escapes or aliases another capture');
  return directory;
}
async function safeRead(dir:string,name:string,optional=false){
  try{
    const file=await fs.realpath(path.join(dir,name));
    if(path.dirname(file).toLowerCase()!==dir.toLowerCase())throw Error('Capture file escapes its directory');
    return await fs.readFile(file,'utf8');
  }catch(error){if(optional&&(error as NodeJS.ErrnoException).code==='ENOENT')return '';throw error;}
}
function parseContinuity(raw:string,indexSha:string,sourceId:string):ContinuityAudit|null{
  if(!raw)return null;
  const value=JSON.parse(raw) as ContinuityAudit;
  if(value.schema_version!==1||value.reviewed!==true||value.index_sha256!==indexSha||value.source_id!==sourceId)return null;
  if(!Array.isArray(value.source_frame_ranges)||value.source_frame_ranges.some(range=>!Array.isArray(range)||range.length!==2||!Number.isSafeInteger(range[0])||!Number.isSafeInteger(range[1])||range[0]<1||range[1]<range[0]))throw Error('Invalid source continuity ranges');
  return value;
}
export async function GET(req:NextRequest){
  try{
    localReviewOrigin(req);
    let base:string;
    try{base=await fs.realpath(root);}catch(error){if((error as NodeJS.ErrnoException).code!=='ENOENT')throw error;const registry=emptyTrainingRegistry();return NextResponse.json({registry,revision:hash(''),clips:[],readiness:trainingReadiness(registry,[])},{headers:{'Cache-Control':'no-store'}});}
    const selected=req.nextUrl.searchParams.get('clip');
    if(selected){
      const directory=await clipDirectory(base,selected),manifest=JSON.parse(await safeRead(directory,'capture.json')),index=await safeRead(directory,'frames.jsonl'),sourceId=manifest.original_sha256??'capture:'+selected;
      const continuityRaw=await safeRead(directory,'continuity_audit.json',true);
      return NextResponse.json({clip:selected,index_sha256:hash(index),source_id:sourceId,continuity:parseContinuity(continuityRaw,hash(index),sourceId),continuity_revision:hash(continuityRaw)},{headers:{'Cache-Control':'no-store'}});
    }
    const {registry,revision}=await readRegistry(base),entries=await fs.readdir(base,{withFileTypes:true});
    const records:ReadinessCapture[]=[],clips=[],metadata=[];
    for(const entry of entries.filter(value=>value.isDirectory()&&validCaptureName(value.name)).sort((a,b)=>a.name.localeCompare(b.name))){
      const directory=await clipDirectory(base,entry.name),manifest=JSON.parse(await safeRead(directory,'capture.json'));
      metadata.push({clip:entry.name,directory,manifest,original_sha256:manifest.original_sha256});
    }
    const sourceConflicts=wholeSourceConflicts(registry,metadata);
    for(const {clip,directory,manifest}of metadata){
      const assignment=registry.captures[clip],session=assignment?registry.sessions[assignment.session_id]:undefined;
      clips.push({clip,original_name:typeof manifest.original_path==='string'?path.basename(manifest.original_path):clip,original_session_id:typeof manifest.session_id==='string'?manifest.session_id:null,assignment:assignment??null,sealed:session?.sealed??false});
      if(session?.role==='test'||sourceConflicts.has(manifest.original_sha256)){
        // Human annotation uses the separate per-clip review route. Automated
        // collection readiness never opens sealed labels, indices or images.
        records.push({clip,original_sha256:manifest.original_sha256,original_session_id:manifest.session_id,frames:[],annotations:{}});
        continue;
      }
      const rawIndex=await safeRead(directory,'frames.jsonl'),frames=rawIndex.split('\n').slice(0,-1).filter(Boolean).map(line=>JSON.parse(line));
      const annotationsRaw=await safeRead(directory,'annotations.json',true),annotations=annotationsRaw?JSON.parse(annotationsRaw):{};
      if(!annotations||typeof annotations!=='object'||Array.isArray(annotations))throw Error('Invalid annotations');
      const readyFrames=[];
      for(const frame of frames){
        if(!/^frame_[0-9]+\.jpg$/.test(frame.file)||!Number.isSafeInteger(frame.source_frame)||frame.source_frame<1)throw Error('Invalid capture frame index');
        const image=await fs.realpath(path.join(directory,frame.file));
        if(path.dirname(image).toLowerCase()!==directory.toLowerCase())throw Error('Image escapes capture directory');
        readyFrames.push({...frame,image_sha256:hash(await fs.readFile(image))});
      }
      const continuityRaw=await safeRead(directory,'continuity_audit.json',true),sourceId=manifest.original_sha256??'capture:'+clip;
      let continuity=parseContinuity(continuityRaw,hash(rawIndex),sourceId);
      if(continuity?.image_sha256&&readyFrames.some(frame=>continuity!.source_frame_ranges.some(([start,end])=>start<=frame.source_frame&&frame.source_frame<=end)&&continuity!.image_sha256![frame.file]!==frame.image_sha256))continuity=null;
      records.push({clip,original_sha256:manifest.original_sha256,original_session_id:manifest.session_id,frames:readyFrames,annotations,continuity});
    }
    const absence=await auditedAbsenceIntervals(base,registry),readiness=trainingReadiness(registry,records,absence.intervals);
    readiness.limitations.push(...absence.issues);
    return NextResponse.json({registry,revision,clips,readiness},{headers:{'Cache-Control':'no-store'}});
  }catch(error){console.error('[training-sessions]',error);return NextResponse.json({error:'Cannot audit local training sessions. Check the server log.'},{status:400});}
}
export async function POST(req:NextRequest){
  let lock:fs.FileHandle|undefined,lockPath:string|undefined;
  try{
    localReviewOrigin(req);
    if(!sameOriginReview(req))return NextResponse.json({error:'Same-origin review required'},{status:403});
    const rawBody=await req.text();if(rawBody.length>10000)throw Error('Session assignment is too large');
    const body=JSON.parse(rawBody);
    const base=await fs.realpath(root),directory=await clipDirectory(base,body.clip),manifest=JSON.parse(await safeRead(directory,'capture.json'));
    if(body.action!==undefined){
      if(body.action!=='confirm_burst_continuity'||body.confirm_source_continuity!==true||body.confirm_capture_complete!==true)throw Error('Explicit completed-capture and viewed-burst continuity confirmation required');
      const indexRaw=await safeRead(directory,'frames.jsonl'),indexSha=hash(indexRaw),sourceId=manifest.original_sha256??'capture:'+body.clip;
      if(!indexRaw.endsWith('\n')||body.index_sha256!==indexSha||body.source_id!==sourceId)return NextResponse.json({error:'Capture source or index changed. Refresh and inspect the burst again.'},{status:409});
      if(!Number.isSafeInteger(body.burst_id)||body.burst_id<1)throw Error('Choose an indexed burst');
      const index=indexRaw.split('\n').slice(0,-1).filter(Boolean).map(line=>JSON.parse(line)),burst=index.filter(frame=>frame.burst_id===body.burst_id);
      if(burst.length<3||burst.length>9||burst.length!==burst[0].burst_length||burst[0].burst_index!==0||burst.some((frame,i)=>!/^frame_[0-9]+\.jpg$/.test(frame.file)||!Number.isSafeInteger(frame.source_frame)||frame.source_frame<1||frame.burst_length!==burst.length||frame.burst_index!==i||!Number.isFinite(frame.elapsed_ms)||frame.elapsed_ms<0||(i>0&&(frame.source_frame!==burst[i-1].source_frame+1||frame.elapsed_ms<=burst[i-1].elapsed_ms||frame.elapsed_ms-burst[i-1].elapsed_ms>100))))throw Error('Only a complete, consecutive burst with increasing source times can be confirmed');
      if(body.start_source_frame!==burst[0].source_frame||body.end_source_frame!==burst.at(-1).source_frame)throw Error('The selected burst range changed');
      const imageHashes:Record<string,string>={};
      for(const frame of burst){const image=await fs.realpath(path.join(directory,frame.file));if(path.dirname(image).toLowerCase()!==directory.toLowerCase())throw Error('Image escapes capture directory');imageHashes[frame.file]=hash(await fs.readFile(image));}
      if(new Set(Object.values(imageHashes)).size!==burst.length)throw Error('Repeated images cannot establish temporal continuity');
      lockPath=path.join(directory,'.continuity-review.lock');
      try{lock=await fs.open(lockPath,'wx');}catch(error){if((error as NodeJS.ErrnoException).code==='EEXIST')return NextResponse.json({error:'Another burst audit is saving. Retry shortly.'},{status:409});throw error;}
      const previousRaw=await safeRead(directory,'continuity_audit.json',true);
      if(body.continuity_revision!==hash(previousRaw))return NextResponse.json({error:'Continuity review changed. Refresh before confirming.'},{status:409});
      const previous=parseContinuity(previousRaw,indexSha,sourceId);
      if(previousRaw&&!previous)return NextResponse.json({error:'A previous audit belongs to a changed source or index. Resolve that capture through a fresh import; no old audit was overwritten.'},{status:409});
      if(await safeRead(directory,'frames.jsonl')!==indexRaw)return NextResponse.json({error:'Capture is still changing. Stop recording and refresh before confirming.'},{status:409});
      const ranges=previous?.source_frame_ranges??[],range:[number,number]=[burst[0].source_frame,burst.at(-1).source_frame];
      const next:ContinuityAudit={schema_version:1,reviewed:true,index_sha256:indexSha,source_id:sourceId,source_frame_ranges:ranges.some(value=>value[0]===range[0]&&value[1]===range[1])?ranges:[...ranges,range].sort((a,b)=>a[0]-b[0]),reviewed_at:new Date().toISOString(),method:'manual_saved_burst',image_sha256:{...previous?.image_sha256,...imageHashes}};
      const history=path.join(directory,'continuity_history');await fs.mkdir(history,{recursive:true});const resolved=await fs.realpath(history);if(path.dirname(resolved).toLowerCase()!==directory.toLowerCase())throw Error('Continuity history escapes capture directory');
      try{await fs.writeFile(path.join(resolved,hash(previousRaw)+'.json'),previousRaw||'null',{flag:'wx'});}catch(error){if((error as NodeJS.ErrnoException).code!=='EEXIST')throw error;}
      const serialized=JSON.stringify(next,null,2)+'\n',temporary=path.join(directory,randomUUID()+'.continuity.tmp');await fs.writeFile(temporary,serialized,{flag:'wx'});await fs.rename(temporary,path.join(directory,'continuity_audit.json'));
      return NextResponse.json({saved:true,continuity:next,continuity_revision:hash(serialized)});
    }
    if(body.confirm_assignment!==true)throw Error('Confirm the whole-session assignment explicitly');
    lockPath=path.join(base,'.training-sessions.lock');
    try{lock=await fs.open(lockPath,'wx');}catch(error){if((error as NodeJS.ErrnoException).code==='EEXIST')return NextResponse.json({error:'Another session assignment is saving. Retry shortly.'},{status:409});throw error;}
    const current=await readRegistry(base);
    if(body.revision!==current.revision)return NextResponse.json({error:'Session assignments changed. Refresh before saving.'},{status:409});
    const next=assignTrainingCapture(current.registry,body,manifest.session_id,new Date().toISOString(),manifest.original_sha256);
    const sourceMetadata=[];
    for(const [clip,assignment]of Object.entries(next.captures)){
      if(next.sessions[assignment.session_id].sealed)continue; // sealed identity is saved registry metadata only
      const dir=await clipDirectory(base,clip),capture=JSON.parse(await safeRead(dir,'capture.json'));
      sourceMetadata.push({clip,original_sha256:capture.original_sha256});
    }
    if(wholeSourceConflicts(next,sourceMetadata).size)throw Error('One original recording is assigned to conflicting sessions, roles or devices. Resolve its audited provenance; no assignments were saved.');
    const history=path.join(base,'training_session_history');await fs.mkdir(history,{recursive:true});
    const resolvedHistory=await fs.realpath(history);if(path.dirname(resolvedHistory).toLowerCase()!==base.toLowerCase())throw Error('Session history escapes training directory');
    try{await fs.writeFile(path.join(resolvedHistory,current.revision+'.json'),current.raw||JSON.stringify(emptyTrainingRegistry()),{flag:'wx'});}catch(error){if((error as NodeJS.ErrnoException).code!=='EEXIST')throw error;}
    const serialized=JSON.stringify(next,null,2)+'\n',temporary=path.join(base,randomUUID()+'.sessions.tmp');
    await fs.writeFile(temporary,serialized,{flag:'wx'});await fs.rename(temporary,path.join(base,registryFile));
    return NextResponse.json({saved:true,registry:next,revision:hash(serialized),assignment:next.captures[body.clip]});
  }catch(error){console.error('[training-sessions]',error);return NextResponse.json({error:error instanceof Error?error.message:'Cannot save this training session assignment'},{status:400});}
  finally{if(lock){await lock.close();if(lockPath)await fs.unlink(lockPath);}}
}
