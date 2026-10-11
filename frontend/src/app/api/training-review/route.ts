import { NextRequest, NextResponse } from 'next/server';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import {validateGeometry,ReviewGeometry,jpegDimensions,createBlurStreak} from '../../../lib/reviewGeometry';
import {localReviewOrigin,sameOriginReview} from '../../../lib/reviewOrigin';
import {previewAbsenceInterval,reviewedAbsence,IndexedReviewFrame,AbsenceProvenance} from '../../../lib/reviewAbsence';
import {validateTrainingRegistry} from '../../../lib/trainingSessions';

export const runtime='nodejs';
export const dynamic='force-dynamic';
const root=process.env.TT_TRAINING_ROOT?path.resolve(process.env.TT_TRAINING_ROOT):path.resolve(process.cwd(),'..','training_data','live');
const classes=['ball','person','table'] as const;
const validClip=(clip:string)=>/^(?:live_[0-9]+_[0-9]+|phone_[0-9a-f]{16}_[0-9]+)$/.test(clip);
type Box=[number,number,number,number];
type Annotation={reviewed:boolean;reviewed_classes?:Record<typeof classes[number],boolean>;ball:Box[];person:Box[];table:Box[];reviewed_at?:string;geometry?:ReviewGeometry;source_frame?:number;elapsed_ms?:number};
const revision=(text:string)=>createHash('sha256').update(text).digest('hex');
async function directory(clip:string){
  if(!validClip(clip))throw Error('Invalid clip');
  const base=await fs.realpath(root),resolved=await fs.realpath(path.join(root,clip));
  if(path.dirname(resolved).toLowerCase()!==base.toLowerCase())throw Error('Clip escapes capture directory');
  return resolved;
}
async function readAnnotations(dir:string){
  try {return await fs.readFile(path.join(dir,'annotations.json'),'utf8');}
  catch(error){if((error as NodeJS.ErrnoException).code==='ENOENT')return '{}';throw error;}
}
async function frameIndex(dir:string){
  const text=await fs.readFile(path.join(dir,'frames.jsonl'),'utf8');
  // The recorder may be appending its last line. Only complete records exist.
  return text.split('\n').slice(0,-1).filter(Boolean).map(line=>JSON.parse(line));
}
function localRequest(req:NextRequest){
  localReviewOrigin(req);
}
export async function GET(req:NextRequest){
  try{
    localRequest(req);const clip=req.nextUrl.searchParams.get('clip');
    if(!clip){
      let entries;try{entries=await fs.readdir(root,{withFileTypes:true});}catch(e){if((e as NodeJS.ErrnoException).code==='ENOENT')return NextResponse.json({clips:[]});throw e;}
      const clips=entries.filter(e=>e.isDirectory()&&validClip(e.name)).map(e=>e.name).sort().reverse();
      const names:Record<string,string>={};
      for(const name of clips.filter(c=>c.startsWith('phone_'))){
        const dir=await directory(name),manifest=JSON.parse(await fs.readFile(path.join(dir,'capture.json'),'utf8'));
        if(typeof manifest.original_path==='string')names[name]=path.basename(manifest.original_path)+(typeof manifest.capture_note==='string'&&manifest.capture_note?' · '+manifest.capture_note:'');
      }
      return NextResponse.json({clips,clip_names:names});
    }
    const dir=await directory(clip),file=req.nextUrl.searchParams.get('file');
    if(file){
      if(!/^frame_[0-9]+\.jpg$/.test(file))throw Error('Invalid frame');
      const resolved=await fs.realpath(path.join(dir,file));
      if(path.dirname(resolved).toLowerCase()!==dir.toLowerCase())throw Error('Frame escapes clip');
      return new NextResponse(new Uint8Array(await fs.readFile(resolved)),{headers:{'Content-Type':'image/jpeg','Cache-Control':'no-store'}});
    }
    const raw=await readAnnotations(dir);
    let capture=null;
    try { const manifest=JSON.parse(await fs.readFile(path.join(dir,'capture.json'),'utf8')); capture={source:manifest.source,original_name:typeof manifest.original_path==='string'?path.basename(manifest.original_path):undefined,session_id:manifest.session_id,source_fps_average:manifest.source_fps_average,frame_count_discrepancy:manifest.frame_count_discrepancy,capture_note:manifest.capture_note,full_source_decode_counted:manifest.full_source_decode_counted,source_frames:manifest.source_frames}; }
    catch(error){if((error as NodeJS.ErrnoException).code!=='ENOENT')throw error;}
    let sealedTest=false;
    try{
      const base=await fs.realpath(root),registryPath=await fs.realpath(path.join(base,'training_sessions.json'));
      if(path.dirname(registryPath).toLowerCase()!==base.toLowerCase())throw Error('Session registry escapes training directory');
      const registry=validateTrainingRegistry(JSON.parse(await fs.readFile(registryPath,'utf8'))),assignment=registry.captures[clip];
      sealedTest=!!assignment&&registry.sessions[assignment.session_id].sealed;
    }catch(error){if((error as NodeJS.ErrnoException).code!=='ENOENT')throw error;}
    const frames=await frameIndex(dir);
    // Human blind annotation can read the chosen test clip, but receives no
    // model-generated box suggestions. Test collection readiness stays sealed.
    return NextResponse.json({frames:sealedTest?frames.map(frame=>({...frame,review_metadata:frame.review_metadata?{...frame.review_metadata,suggestions:undefined}:undefined})):frames,annotations:JSON.parse(raw),revision:revision(raw),capture,sealed_test:sealedTest});
  }catch(error){console.error('[training-review]',error);return NextResponse.json({error:'Cannot read this local capture. Check the server log.'},{status:400});}
}
export async function POST(req:NextRequest){
  let lock:fs.FileHandle|undefined;
  let lockPath:string|undefined;
  try{
    localRequest(req);
    if(!sameOriginReview(req))return NextResponse.json({error:'Same-origin review required'},{status:403});
    const rawBody=await req.text();if(rawBody.length>100_000)throw Error('Review too large');
    const body=JSON.parse(rawBody),dir=await directory(body.clip);
    if(body.action!==undefined){
      if(!['preview_ball_absence_interval','confirm_ball_absence_interval'].includes(body.action))throw Error('Invalid review action');
      const captureRaw=await fs.readFile(path.join(dir,'capture.json'),'utf8'),capture=JSON.parse(captureRaw);
      if(capture.source!=='recorded_camera'||capture.full_source_decode_counted!==true||!Number.isSafeInteger(capture.source_frames)||capture.source_frames<1||!Number.isSafeInteger(capture.sampled_frames)||capture.sampled_frames<1||!(/^[0-9a-f]{64}$/).test(capture.original_sha256)||typeof capture.session_id!=='string'||!capture.session_id.trim())throw Error('Bulk absence review requires a completed recording import with source and session identity.');
      const indexRaw=await fs.readFile(path.join(dir,'frames.jsonl'),'utf8');
      if(!indexRaw.endsWith('\n')||indexRaw.split('\n').slice(0,-1).some(line=>!line.trim()))throw Error('Capture index is incomplete. Finish importing before bulk review.');
      const frames:IndexedReviewFrame[]=indexRaw.split('\n').slice(0,-1).map(line=>JSON.parse(line));
      if(frames.length!==capture.sampled_frames)throw Error('Capture index count differs from the completed import.');
      if(frames.some(frame=>frame.source_frame>capture.source_frames))throw Error('Capture index exceeds the decoded source frames.');
      if(body.end_source_frame>capture.source_frames)throw Error('Interval exceeds the decoded source frames.');
      if(body.action==='confirm_ball_absence_interval'){
        if(body.confirm_absence!==true)throw Error('Explicit no-ball confirmation is required.');
        lockPath=path.join(dir,'.review.lock');
        try{lock=await fs.open(lockPath,'wx');}catch(e){if((e as NodeJS.ErrnoException).code==='EEXIST')return NextResponse.json({error:'Another review is being saved; retry shortly.'},{status:409});throw e;}
      }
      const raw=await readAnnotations(dir);
      if(body.revision!==revision(raw))return NextResponse.json({error:'Annotations changed. Reload this clip before bulk review.'},{status:409});
      const annotations=JSON.parse(raw),interval={start_source_frame:body.start_source_frame,end_source_frame:body.end_source_frame};
      if(!annotations||typeof annotations!=='object'||Array.isArray(annotations))throw Error('Invalid saved annotation dictionary.');
      const preview=previewAbsenceInterval(frames,annotations,interval);
      const images:Record<string,string>={};
      for(const frame of preview.selected){
        const imagePath=await fs.realpath(path.join(dir,frame.file));
        if(path.dirname(imagePath).toLowerCase()!==dir.toLowerCase())throw Error('Frame escapes clip');
        const image=await fs.readFile(imagePath);jpegDimensions(image);
        images[frame.file]=createHash('sha256').update(image).digest('hex');
      }
      const token=revision(JSON.stringify({clip:body.clip,interval,annotation:revision(raw),capture:revision(captureRaw),index:revision(indexRaw),files:preview.selected,images}));
      if(body.action==='preview_ball_absence_interval')return NextResponse.json({preview:true,preview_token:token,revision:revision(raw),interval,affected_frames:preview.selected.length,frames:preview.selected,conflicts:preview.conflicts,original_sha256:capture.original_sha256,session_id:capture.session_id});
      if(body.preview_token!==token)return NextResponse.json({error:'The interval or capture changed. Preview it again before confirming.'},{status:409});
      if(preview.conflicts.length)return NextResponse.json({error:'Existing visible or hidden ball evidence conflicts with absence. Resolve those frames individually.',conflicts:preview.conflicts},{status:409});
      if(await fs.readFile(path.join(dir,'capture.json'),'utf8')!==captureRaw||await fs.readFile(path.join(dir,'frames.jsonl'),'utf8')!==indexRaw)return NextResponse.json({error:'Capture changed while saving. Preview the interval again.'},{status:409});
      const provenance:AbsenceProvenance={...interval,method:'manual_interval',review_id:randomUUID(),reviewed_at:new Date().toISOString(),original_sha256:capture.original_sha256,session_id:capture.session_id,index_sha256:revision(indexRaw),previous_annotation_revision:revision(raw),sampled_frames:preview.selected.length};
      for(const frame of preview.selected)annotations[frame.file]=reviewedAbsence(annotations[frame.file],frame,provenance,images[frame.file]);
      const history=path.join(dir,'annotation_history');await fs.mkdir(history,{recursive:true});
      try{await fs.writeFile(path.join(history,revision(raw)+'.json'),raw,{flag:'wx'});}catch(e){if((e as NodeJS.ErrnoException).code!=='EEXIST')throw e;}
      const next=JSON.stringify(annotations,null,2),temporary=path.join(dir,randomUUID()+'.review.tmp');
      await fs.writeFile(temporary,next,{flag:'wx'});await fs.rename(temporary,path.join(dir,'annotations.json'));
      return NextResponse.json({saved:true,revision:revision(next),annotations,affected_frames:preview.selected.length,interval_review:provenance});
    }
    if(body.scope!==undefined&&!['ball','all'].includes(body.scope))throw Error('Invalid review scope');
    const selected=body.scope==='ball'?['ball'] as const:classes;
    if(!/^frame_[0-9]+\.jpg$/.test(body.file))throw Error('Invalid frame');
    const frames=await frameIndex(dir);if(!frames.some(f=>f.file===body.file))throw Error('Frame is not indexed');
    const labels:Annotation={reviewed:body.reviewed===true,ball:[],person:[],table:[]};
    for(const cls of classes){
      if(!Array.isArray(body.labels?.[cls])||body.labels[cls].length>100)throw Error('Explicit labels are required for all classes');
      labels[cls]=body.labels[cls].map((box:unknown)=>{
        if(!Array.isArray(box)||box.length!==4||box.some(v=>typeof v!=='number'||!Number.isFinite(v)))throw Error('Invalid bounding box');
        const [x,y,w,h]=box;if(x<0||y<0||w<=0||h<=0||x+w>1.000001||y+h>1.000001)throw Error('Box outside image');
        return box as Box;
      });
    }
    labels.reviewed_at=new Date().toISOString();
    labels.geometry=validateGeometry(body.labels.geometry);
    if(labels.geometry.blur_streak){
      if(!['visible','partial'].includes(labels.geometry.visibility))throw Error('Blur endpoints require visible or partly visible ball evidence');
      const imagePath=await fs.realpath(path.join(dir,body.file));
      if(path.dirname(imagePath).toLowerCase()!==dir.toLowerCase())throw Error('Frame escapes clip');
      const size=jpegDimensions(await fs.readFile(imagePath)),streak=labels.geometry.blur_streak;
      if(streak.source_width!==size.width||streak.source_height!==size.height)throw Error('Blur source dimensions changed. Reload the frame before saving.');
      labels.geometry.blur_streak=createBlurStreak(streak.endpoints[0],streak.endpoints[1],size,body.reviewed===true);
    }
    if(labels.reviewed&&labels.geometry.visibility==='absent'&&labels.ball.length)throw Error('Absent-ball tag conflicts with reviewed ball boxes');
    const sourceFrame=frames.find(f=>f.file===body.file)!;
    labels.source_frame=sourceFrame.source_frame;
    labels.elapsed_ms=sourceFrame.elapsed_ms;
    lockPath=path.join(dir,'.review.lock');
    try{lock=await fs.open(lockPath,'wx');}catch(e){if((e as NodeJS.ErrnoException).code==='EEXIST')return NextResponse.json({error:'Another review is being saved; retry shortly.'},{status:409});throw e;}
    const raw=await readAnnotations(dir);
    if(body.revision!==revision(raw))return NextResponse.json({error:'Annotations changed. Reload this clip before saving.'},{status:409});
    const annotations=JSON.parse(raw),previous=annotations[body.file] as Annotation|undefined;
    if(body.scope!==undefined){
      const flags={ball:previous?.reviewed_classes?.ball??previous?.reviewed===true,
                   person:previous?.reviewed_classes?.person??previous?.reviewed===true,
                   table:previous?.reviewed_classes?.table??previous?.reviewed===true};
      for(const cls of selected)flags[cls]=body.reviewed===true;
      if(body.scope==='ball'){
        labels.person=previous?.person??[];labels.table=previous?.table??[];
        const oldGeometry=previous?.geometry;
        if(oldGeometry)labels.geometry={...oldGeometry,visibility:labels.geometry.visibility,blur:labels.geometry.blur,blur_streak:labels.geometry.blur_streak,
          ...(labels.geometry.rally_id!==undefined?{rally_id:labels.geometry.rally_id}:{}),...(labels.geometry.rally_reviewed!==undefined?{rally_reviewed:labels.geometry.rally_reviewed}:{}),...(labels.geometry.quality_tags!==undefined?{quality_tags:labels.geometry.quality_tags}:{})};
      }
      if(flags.ball&&labels.geometry.visibility==='absent'&&labels.ball.length)throw Error('Absent-ball tag conflicts with reviewed ball boxes');
      if(body.reviewed===true&&labels.geometry.visibility==='occluded'&&labels.ball.length)throw Error('A fully occluded ball cannot have an observed box');
      if(body.reviewed===true&&!labels.ball.length&&!labels.geometry.blur_streak&&['visible','partial'].includes(labels.geometry.visibility))throw Error('Visible ball needs a box or explicitly reviewed streak. Use absent or hidden/unknown only when appropriate.');
      if(body.reviewed===true&&!labels.ball.length&&labels.geometry.visibility==='unknown')throw Error('An empty ball label needs explicit absent or occluded visibility.');
      labels.reviewed_classes=flags;labels.reviewed=Object.values(flags).some(Boolean);
    }
    annotations[body.file]=labels;
    // Preserve immutable prior revisions as well as the current optimistic lock.
    const history=path.join(dir,'annotation_history');await fs.mkdir(history,{recursive:true});
    const historical=path.join(history,revision(raw)+'.json');
    try{await fs.writeFile(historical,raw,{flag:'wx'});}catch(e){if((e as NodeJS.ErrnoException).code!=='EEXIST')throw e;}
    const next=JSON.stringify(annotations,null,2),temporary=path.join(dir,randomUUID()+'.review.tmp');
    await fs.writeFile(temporary,next,{flag:'wx'});await fs.rename(temporary,path.join(dir,'annotations.json'));
    return NextResponse.json({saved:true,revision:revision(next),annotation:labels});
  }catch(error){console.error('[training-review]',error);return NextResponse.json({error:error instanceof Error?error.message:'Review could not be saved'},{status:400});}
  finally{if(lock){await lock.close();await fs.unlink(lockPath!);}}
}
