import { NextRequest, NextResponse } from 'next/server';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import {validateGeometry,ReviewGeometry} from '../../../lib/reviewGeometry';

export const runtime='nodejs';
export const dynamic='force-dynamic';
const root=process.env.TT_TRAINING_ROOT?path.resolve(process.env.TT_TRAINING_ROOT):path.resolve(process.cwd(),'..','training_data','live');
const classes=['ball','person','table'] as const;
type Box=[number,number,number,number];
type Annotation={reviewed:boolean;ball:Box[];person:Box[];table:Box[];reviewed_at?:string;geometry?:ReviewGeometry;source_frame?:number;elapsed_ms?:number};
const revision=(text:string)=>createHash('sha256').update(text).digest('hex');
async function directory(clip:string){
  if(!/^live_[0-9]+_[0-9]+$/.test(clip))throw Error('Invalid clip');
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
  const host=req.nextUrl.hostname;
  if(!['localhost','127.0.0.1','[::1]'].includes(host))throw Error('Training review is available on localhost only');
}
export async function GET(req:NextRequest){
  try{
    localRequest(req);const clip=req.nextUrl.searchParams.get('clip');
    if(!clip){
      let entries;try{entries=await fs.readdir(root,{withFileTypes:true});}catch(e){if((e as NodeJS.ErrnoException).code==='ENOENT')return NextResponse.json({clips:[]});throw e;}
      return NextResponse.json({clips:entries.filter(e=>e.isDirectory()&&/^live_[0-9]+_[0-9]+$/.test(e.name)).map(e=>e.name).sort().reverse()});
    }
    const dir=await directory(clip),file=req.nextUrl.searchParams.get('file');
    if(file){
      if(!/^frame_[0-9]+\.jpg$/.test(file))throw Error('Invalid frame');
      const resolved=await fs.realpath(path.join(dir,file));
      if(path.dirname(resolved).toLowerCase()!==dir.toLowerCase())throw Error('Frame escapes clip');
      return new NextResponse(new Uint8Array(await fs.readFile(resolved)),{headers:{'Content-Type':'image/jpeg','Cache-Control':'no-store'}});
    }
    const raw=await readAnnotations(dir);
    return NextResponse.json({frames:await frameIndex(dir),annotations:JSON.parse(raw),revision:revision(raw)});
  }catch(error){console.error('[training-review]',error);return NextResponse.json({error:'Cannot read this local capture. Check the server log.'},{status:400});}
}
export async function POST(req:NextRequest){
  let lock:fs.FileHandle|undefined;
  let lockPath:string|undefined;
  try{
    localRequest(req);
    if(req.headers.get('origin')!==req.nextUrl.origin)return NextResponse.json({error:'Same-origin review required'},{status:403});
    const rawBody=await req.text();if(rawBody.length>100_000)throw Error('Review too large');
    const body=JSON.parse(rawBody),dir=await directory(body.clip);
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
    if(labels.reviewed&&labels.geometry.visibility==='absent'&&labels.ball.length)throw Error('Absent-ball tag conflicts with reviewed ball boxes');
    const sourceFrame=frames.find(f=>f.file===body.file)!;
    labels.source_frame=sourceFrame.source_frame;
    labels.elapsed_ms=sourceFrame.elapsed_ms;
    lockPath=path.join(dir,'.review.lock');
    try{lock=await fs.open(lockPath,'wx');}catch(e){if((e as NodeJS.ErrnoException).code==='EEXIST')return NextResponse.json({error:'Another review is being saved; retry shortly.'},{status:409});throw e;}
    const raw=await readAnnotations(dir);
    if(body.revision!==revision(raw))return NextResponse.json({error:'Annotations changed. Reload this clip before saving.'},{status:409});
    const annotations=JSON.parse(raw);annotations[body.file]=labels;
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
