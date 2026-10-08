'use client';
import {useEffect,useState} from 'react';
import {landmarkNames,eventNames,emptyGeometry,LandmarkName,ReviewGeometry} from '../../lib/reviewGeometry';
import {ReviewFrameImage} from '../../components/ReviewFrameImage';
type Kind='ball'|'person'|'table';
type Box=[number,number,number,number];
type Labels=Record<Kind,Box[]>&{geometry?:ReviewGeometry};
type Frame={file:string;source_frame:number;review_metadata?:{reasons?:string[];suggestions?:Labels;ball_model_sha256?:string}};
type Capture={source?:string;original_name?:string;session_id?:string;source_fps_average?:number;frame_count_discrepancy?:number;capture_note?:string};
const empty=():Labels=>({ball:[],person:[],table:[]});
export default function TrainingReview(){
  const [clips,setClips]=useState<string[]>([]),[clip,setClip]=useState(''),[frames,setFrames]=useState<Frame[]>([]),[index,setIndex]=useState(0);
  const [drafts,setDrafts]=useState<Record<string,Labels>>({}),[annotations,setAnnotations]=useState<Record<string,Labels&{reviewed:boolean}>>({});
  const [revision,setRevision]=useState(''),[message,setMessage]=useState(''),[kind,setKind]=useState<Kind>('ball'),[busy,setBusy]=useState(false);
  const [capture,setCapture]=useState<Capture|null>(null);
  const [clipNames,setClipNames]=useState<Record<string,string>>({});
  const [landmark,setLandmark]=useState<LandmarkName|''>('');
  const frame=frames[index];
  const key=clip+'/'+(frame?.file??''),labels=drafts[key]??annotations[frame?.file??'']??empty();
  function setLabels(value:Labels|((previous:Labels)=>Labels)){
    setDrafts(previous=>({...previous,[key]:typeof value==='function'?value(previous[key]??annotations[frame?.file??'']??empty()):value}));
  }
  useEffect(()=>{
    if(!Object.keys(drafts).length)return;
    const warn=(event:BeforeUnloadEvent)=>{event.preventDefault();event.returnValue='';};
    window.addEventListener('beforeunload',warn);return()=>window.removeEventListener('beforeunload',warn);
  },[drafts]);
  useEffect(()=>{fetch('/api/training-review').then(r=>r.json()).then(d=>{if(d.error)throw Error(d.error);setClips(d.clips);setClipNames(d.clip_names??{});}).catch(e=>setMessage(e.message));},[]);
  async function load(value:string){
    setBusy(true);setMessage('');
    try{const r=await fetch('/api/training-review?clip='+encodeURIComponent(value)),d=await r.json();if(!r.ok)throw Error(d.error);
      setClip(value);setFrames(d.frames);setAnnotations(d.annotations);setRevision(d.revision);setIndex(0);setCapture(d.capture??null);
    }catch(e){setMessage(String(e));}finally{setBusy(false);}
  }
  function finish(a:[number,number],b:[number,number],size:{width:number;height:number}){
    if(busy)return;
    if(landmark){setLabels(previous=>({...previous,geometry:{...(previous.geometry??emptyGeometry()),landmarks:{...previous.geometry?.landmarks,[landmark]:b},landmarks_reviewed:false}}));return;}
    let box:Box=[Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.abs(a[0]-b[0]),Math.abs(a[1]-b[1])];
    if(kind==='ball'&&box[2]<.003&&box[3]<.003){const w=Math.min(1,12/size.width),h=Math.min(1,12/size.height);box=[Math.max(0,Math.min(1-w,b[0]-w/2)),Math.max(0,Math.min(1-h,b[1]-h/2)),w,h];}
    if(box[2]>0&&box[3]>0)setLabels(previous=>({...previous,[kind]:[...previous[kind],box]}));
  }
  async function save(reviewed:boolean){
    if(!frame)return;setBusy(true);setMessage('');
    try{const r=await fetch('/api/training-review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({clip,file:frame.file,labels,reviewed,revision})}),d=await r.json();if(!r.ok)throw Error(d.error);
      setRevision(d.revision);setAnnotations(a=>({...a,[frame.file]:d.annotation}));setDrafts(a=>{const next={...a};delete next[key];return next;});setMessage(reviewed?'Reviewed labels saved locally.':'Marked unreviewed; excluded from training.');
    }catch(e){setMessage(String(e));}finally{setBusy(false);}
  }
  return <main className="min-h-screen bg-slate-950 text-slate-200 p-6 space-y-5 max-w-6xl mx-auto">
    <a href="/" className="text-cyan-300 text-sm">← Dashboard</a><h1 className="text-2xl font-bold">Training review</h1>
    <p className="text-slate-400 text-sm">Draw boxes around every visible ball, person and table. Click a ball for an initial 12-pixel box, or drag its full extent. Empty class lists mean you verified that class is absent. Predictions remain suggestions until you explicitly save reviewed labels.</p>
    <select aria-label="Capture clip" value={clip} disabled={busy} onChange={e=>load(e.target.value)} className="bg-slate-900 border border-slate-700 rounded-lg p-2"><option value="">Select a captured clip</option>{clips.map(c=><option key={c} value={c}>{clipNames[c]??c}</option>)}</select>
    {!clips.length&&<p>No local training captures yet. Use the live-camera recording controls or import original camera recordings first.</p>}
    {capture?.source==='recorded_camera'&&<div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-4 text-sm space-y-2">
      <p className="text-cyan-300">{capture.original_name} · {capture.source_fps_average?.toFixed(2)} average FPS · session {capture.session_id}</p>
      {capture.capture_note&&<p className="text-amber-200">Review focus: {capture.capture_note}</p>}
      <p className="text-amber-200">Original camera footage imported as unreviewed consecutive-frame bursts. Keep every clip from this session in the same dataset split. These frames do not become training labels until you explicitly review and save them.</p>
      {!!capture.frame_count_discrepancy&&<p className="text-amber-200">Reported and decoded frame counts differ by {capture.frame_count_discrepancy}. Imported frames use sequential decoding and source timestamps; recording playback timing still needs separate verification.</p>}
    </div>}
    {!!Object.keys(drafts).length&&<p className="text-amber-300 text-sm">{Object.keys(drafts).length} frame(s) have unsaved edits. Drafts stay while changing frames or clips; save each before leaving this page.</p>}
    {frame&&<>
      <div className="flex gap-3 flex-wrap items-center font-mono text-xs"><button disabled={busy||index===0} onClick={()=>setIndex(index-1)}>Previous</button><span>{index+1} / {frames.length} · source frame {frame.source_frame} · {annotations[frame.file]?.reviewed?'REVIEWED':'UNREVIEWED'}</span><button disabled={busy||index===frames.length-1} onClick={()=>setIndex(index+1)}>Next</button></div>
      <p className="text-amber-300 text-xs font-mono">{frame.review_metadata?.reasons?.join(' · ')??'Explicit training capture'}</p>
      <ReviewFrameImage enabled={!busy} src={'/api/training-review?clip='+encodeURIComponent(clip)+'&file='+encodeURIComponent(frame.file)} onAnnotation={finish}>
          {(['ball','person','table'] as Kind[]).flatMap(k=>labels[k].map((b,i)=><rect key={k+i} x={b[0]*100} y={b[1]*100} width={b[2]*100} height={b[3]*100} fill="none" stroke={k==='ball'?'#fbbf24':k==='person'?'#a78bfa':'#34d399'} strokeWidth=".25"/>))}
          {Object.entries(labels.geometry?.landmarks??{}).map(([name,p])=><g key={name}><circle cx={p[0]*100} cy={p[1]*100} r=".65" fill="#34d399"/><text x={p[0]*100+1} y={p[1]*100} fontSize="1.8" fill="white">{name}</text></g>)}
      </ReviewFrameImage>
      <div className="flex flex-wrap gap-3 text-sm">{(['ball','person','table'] as Kind[]).map(k=><button key={k} onClick={()=>{setKind(k);setLandmark('');}} className={'px-3 py-2 rounded-lg border '+(kind===k&&!landmark?'border-cyan-400 text-cyan-300':'border-slate-700')}>{k} · {labels[k].length}</button>)}
        <button disabled={busy} onClick={()=>setLabels(l=>({...l,[kind]:l[kind].slice(0,-1)}))}>Undo {kind}</button>
        <button disabled={busy} onClick={()=>setLabels(l=>({...l,[kind]:[]}))}>Clear {kind}</button>
        <button disabled={busy} onClick={()=>setLabels(frame.review_metadata?.suggestions??empty())}>Load model suggestions</button>
      </div>
      <fieldset className="border border-slate-700 rounded-xl p-4 space-y-3 text-sm"><legend>Table geometry and temporal labels</legend>
        <p className="text-slate-400">Near/far refer to the camera; left/right refer to the image. Select a landmark, then click its visible location. Omitted landmarks remain unknown. Return to a box class to draw boxes.</p>
        <select aria-label="Table landmark" value={landmark} onChange={e=>setLandmark(e.target.value as LandmarkName|'')} className="bg-slate-900 p-2"><option value="">Box annotation</option>{landmarkNames.map(n=><option key={n}>{n}</option>)}</select>
        <button className="ml-3 text-amber-300" disabled={!landmark||busy} onClick={()=>setLabels(l=>{const g=l.geometry??emptyGeometry(),points={...g.landmarks};if(landmark)delete points[landmark];return {...l,geometry:{...g,landmarks:points,landmarks_reviewed:false}};})}>Clear selected landmark</button>
        <label className="block"><input type="checkbox" checked={labels.geometry?.landmarks_reviewed??false} onChange={e=>setLabels(l=>({...l,geometry:{...(l.geometry??emptyGeometry()),landmarks_reviewed:e.target.checked}}))}/> I checked the visible landmarks</label>
        <div className="flex flex-wrap gap-4">{(['visibility','blur'] as const).map(field=><label key={field}>{field} <select aria-label={field} className="bg-slate-900 p-2" value={labels.geometry?.[field]??'unknown'} onChange={e=>setLabels(l=>({...l,geometry:{...(l.geometry??emptyGeometry()),[field]:e.target.value}}))}>{(field==='visibility'?['unknown','visible','partial','occluded','absent']:['unknown','none','mild','moderate','severe','extreme']).map(v=><option key={v}>{v}</option>)}</select></label>)}</div>
        <p className="text-slate-400">Events apply at this captured source frame and its recorded elapsed time. A sparse clip may miss the actual contact; leave it unreviewed when uncertain.</p>
        <div className="flex flex-wrap gap-3">{eventNames.map(event=><label key={event}><input type="checkbox" checked={labels.geometry?.events.includes(event)??false} onChange={e=>setLabels(l=>{const g=l.geometry??emptyGeometry();return {...l,geometry:{...g,events:e.target.checked?[...g.events,event]:g.events.filter(x=>x!==event),events_reviewed:false}};})}/> {event}</label>)}</div>
        <label className="block"><input type="checkbox" checked={labels.geometry?.events_reviewed??false} onChange={e=>setLabels(l=>({...l,geometry:{...(l.geometry??emptyGeometry()),events_reviewed:e.target.checked}}))}/> I checked the events at this frame (including no event)</label>
      </fieldset>
      <div className="flex gap-3"><button disabled={busy} onClick={()=>save(true)} className="bg-emerald-400 text-slate-950 font-bold rounded-lg px-4 py-2">Save reviewed labels</button><button disabled={busy} onClick={()=>save(false)} className="border border-slate-700 rounded-lg px-4 py-2">Exclude / mark unreviewed</button></div>
      <p className="text-xs text-slate-500">Review does not assign dataset splits or train/deploy a model. Keep complete recording sessions in separate train, validation and test groups using TRAINING_LIVE.md.</p>
    </>}
    {message&&<p role="status" className="text-amber-300">{message}</p>}
  </main>;
}
