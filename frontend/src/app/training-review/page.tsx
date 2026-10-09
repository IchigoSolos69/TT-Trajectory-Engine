'use client';
import {useEffect,useState} from 'react';
import {landmarkNames,eventNames,emptyGeometry,LandmarkName} from '../../lib/reviewGeometry';
import {ReviewFrameImage} from '../../components/ReviewFrameImage';
import {Kind,Box,Labels,Annotation,Frame,emptyLabels,classReviewed,ballReviewProblem,previousBallSuggestion,nextBallReviewIndex,validDraft} from '../../lib/reviewAnnotations';
type Capture={source?:string;original_name?:string;session_id?:string;source_fps_average?:number;frame_count_discrepancy?:number;capture_note?:string};
const draftStorageKey='tt-training-review-drafts-v1';
export default function TrainingReview(){
  const [clips,setClips]=useState<string[]>([]),[clip,setClip]=useState(''),[frames,setFrames]=useState<Frame[]>([]),[index,setIndex]=useState(0);
  const [drafts,setDrafts]=useState<Record<string,Labels>>({}),[annotations,setAnnotations]=useState<Record<string,Annotation>>({});
  const [revision,setRevision]=useState(''),[message,setMessage]=useState(''),[kind,setKind]=useState<Kind>('ball'),[busy,setBusy]=useState(false);
  const [capture,setCapture]=useState<Capture|null>(null);
  const [clipNames,setClipNames]=useState<Record<string,string>>({});
  const [landmark,setLandmark]=useState<LandmarkName|''>('');
  const [scope,setScope]=useState<'ball'|'all'>('ball'),[draftsRestored,setDraftsRestored]=useState(false);
  const frame=frames[index];
  const key=clip+'/'+(frame?.file??''),labels=drafts[key]??annotations[frame?.file??'']??emptyLabels();
  const problem=ballReviewProblem(labels),previousBall=previousBallSuggestion(frames,index,annotations);
  function setLabels(value:Labels|((previous:Labels)=>Labels)){
    setDrafts(previous=>({...previous,[key]:typeof value==='function'?value(previous[key]??annotations[frame?.file??'']??emptyLabels()):value}));
  }
  useEffect(()=>{
    let active=true;
    const restore=async()=>{
      try{
        const raw=window.ttDesktop?await window.ttDesktop.getReviewDrafts():localStorage.getItem(draftStorageKey);
        const stored=typeof raw==='string'?JSON.parse(raw):raw;
        if(stored){if(typeof stored!=='object'||Array.isArray(stored)||Object.values(stored).some(value=>!validDraft(value)))throw Error('Invalid saved drafts');if(active)setDrafts(stored);}
        if(active)setDraftsRestored(true);
      }catch{if(active)setMessage('Local draft recovery failed. Saved reviewed labels are unaffected. Draft storage was not overwritten.');}
    };
    void restore();return()=>{active=false;};
  },[]);
  useEffect(()=>{
    if(!draftsRestored)return;
    if(window.ttDesktop){window.ttDesktop.saveReviewDrafts(drafts).catch(()=>setMessage('Local draft storage is unavailable. Save edits before leaving this page.'));return;}
    try{localStorage.setItem(draftStorageKey,JSON.stringify(drafts));}
    catch{setMessage('Local draft storage is unavailable. Save edits before leaving this page.');}
  },[drafts,draftsRestored]);
  useEffect(()=>{
    // Desktop drafts are flushed by the main process before application exit.
    // A browser beforeunload prompt would otherwise cancel native window close.
    if(window.ttDesktop)return;
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
    if(kind==='ball'&&box[2]*size.width<2&&box[3]*size.height<2){const w=labels.ball[0]?.[2]??previousBall?.[2]??Math.min(1,12/size.width),h=labels.ball[0]?.[3]??previousBall?.[3]??Math.min(1,12/size.height);box=[Math.max(0,Math.min(1-w,b[0]-w/2)),Math.max(0,Math.min(1-h,b[1]-h/2)),w,h];}
    if(box[2]>0&&box[3]>0)setLabels(previous=>({...previous,[kind]:scope==='ball'?[box]:[...previous[kind],box],geometry:kind==='ball'?{...(previous.geometry??emptyGeometry()),visibility:'visible'}:previous.geometry}));
  }
  function reuseBall(){if(previousBall){setKind('ball');setLandmark('');setLabels(l=>({...l,ball:[previousBall],geometry:{...(l.geometry??emptyGeometry()),visibility:'visible'}}));setMessage('Previous box copied as a draft. Check and move it to the current ball before saving.');}}
  async function save(reviewed:boolean,advance=false){
    if(!frame||busy)return;setBusy(true);setMessage('');
    try{const r=await fetch('/api/training-review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({clip,file:frame.file,labels,reviewed,revision,scope})}),d=await r.json();if(!r.ok)throw Error(d.error);
      setRevision(d.revision);setAnnotations(a=>({...a,[frame.file]:d.annotation}));setDrafts(a=>{const next={...a};delete next[key];return next;});setMessage(reviewed?(scope==='ball'?'Ball label saved. People/table review status preserved.':'Reviewed labels saved locally.'):'Selected labels marked unreviewed; excluded from their training dataset.');
      if(advance&&index<frames.length-1)setIndex(index+1);
    }catch(e){setMessage(String(e));}finally{setBusy(false);}
  }
  useEffect(()=>{
    const shortcut=(event:KeyboardEvent)=>{const target=event.target as HTMLElement;if(busy||!frame||target.isContentEditable||['INPUT','SELECT','TEXTAREA','SUMMARY','BUTTON'].includes(target.tagName))return;
      if(event.key==='Enter'&&!problem){event.preventDefault();void save(true,true);}
      else if(event.key==='ArrowRight'){event.preventDefault();setIndex(i=>Math.min(frames.length-1,i+1));}
      else if(event.key==='ArrowLeft'){event.preventDefault();setIndex(i=>Math.max(0,i-1));}
    };
    window.addEventListener('keydown',shortcut);return()=>window.removeEventListener('keydown',shortcut);
  });
  return <main className="min-h-screen bg-slate-950 text-slate-200 p-6 space-y-5 max-w-6xl mx-auto">
    <a href="/" className="text-cyan-300 text-sm">← Dashboard</a><h1 className="text-2xl font-bold">Training review</h1>
    <p className="text-slate-400 text-sm">Ball-only review: click the ball, adjust its box if needed, then Save & next. You do not need to label people, table corners or events to train the ball detector. A click starts a 12-pixel box or reuses the previous reviewed size; drag to correct a blurred ball’s extent.</p>
    <div className="flex flex-wrap gap-3 text-sm"><button disabled={busy} aria-pressed={scope==='ball'} onClick={()=>{setScope('ball');setKind('ball');setLandmark('');}} className="border border-cyan-500/40 rounded-lg px-3 py-2">Ball only · faster</button><button disabled={busy} aria-pressed={scope==='all'} onClick={()=>setScope('all')} className="border border-slate-700 rounded-lg px-3 py-2">Full scene & advanced</button></div>
    <select aria-label="Capture clip" value={clip} disabled={busy} onChange={e=>load(e.target.value)} className="bg-slate-900 border border-slate-700 rounded-lg p-2"><option value="">Select a captured clip</option>{clips.map(c=><option key={c} value={c}>{clipNames[c]??c}</option>)}</select>
    {!clips.length&&<p>No local training captures yet. Use the live-camera recording controls or import original camera recordings first.</p>}
    {capture?.source==='recorded_camera'&&<div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-4 text-sm space-y-2">
      <p className="text-cyan-300">{capture.original_name} · {capture.source_fps_average?.toFixed(2)} average FPS · session {capture.session_id}</p>
      {capture.capture_note&&<p className="text-amber-200">Review focus: {capture.capture_note}</p>}
      <p className="text-amber-200">Original camera footage imported as unreviewed consecutive-frame bursts. Keep every clip from this session in the same dataset split. These frames do not become training labels until you explicitly review and save them.</p>
      {!!capture.frame_count_discrepancy&&<p className="text-amber-200">Reported and decoded frame counts differ by {capture.frame_count_discrepancy}. Imported frames use sequential decoding and source timestamps; recording playback timing still needs separate verification.</p>}
    </div>}
    {!!Object.keys(drafts).length&&<p className="text-amber-300 text-sm">{Object.keys(drafts).length} frame(s) have unconfirmed drafts stored locally. Drafts stay excluded from training until reviewed and saved.</p>}
    {frame&&<>
      <div className="flex gap-3 flex-wrap items-center font-mono text-xs"><button disabled={busy||index===0} onClick={()=>setIndex(index-1)}>Previous</button><span>{index+1} / {frames.length} · source frame {frame.source_frame} · {classReviewed(annotations[frame.file],'ball')?'BALL REVIEWED':'BALL UNREVIEWED'}</span><button disabled={busy||index===frames.length-1} onClick={()=>setIndex(index+1)}>Next</button><button disabled={busy} onClick={()=>{const next=nextBallReviewIndex(frames,index,annotations);if(next>=0)setIndex(next);else setMessage('No remaining ball frames need review in this capture.');}}>Next needing review</button></div>
      <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/5 p-3 space-y-3 text-sm">
        <div className="flex flex-wrap gap-3"><button disabled={busy||!!problem} onClick={()=>save(true,true)} className="bg-emerald-400 text-slate-950 font-bold rounded-lg px-4 py-2">Save & next</button><button disabled={busy||!!problem} onClick={()=>save(true)} className="border border-slate-700 rounded-lg px-3 py-2">Save this frame</button><button disabled={busy} onClick={()=>save(false,true)} className="border border-slate-700 rounded-lg px-3 py-2">Skip / keep unreviewed</button><button disabled={busy||!previousBall} onClick={reuseBall} className="border border-slate-700 rounded-lg px-3 py-2">Reuse previous ball box</button></div>
        <div className="flex flex-wrap gap-3">{(['visible','partial','absent','occluded'] as const).map((visibility,i)=><button key={visibility} disabled={busy} aria-pressed={labels.geometry?.visibility===visibility} onClick={()=>setLabels(l=>({...l,ball:visibility==='absent'||visibility==='occluded'?[]:l.ball,geometry:{...(l.geometry??emptyGeometry()),visibility}}))} className="border border-slate-700 rounded-lg px-3 py-2">{['Visible','Partly visible','Not in frame','Hidden / unsure'][i]}</button>)}</div>
        {problem&&<p role="alert" className="text-amber-300">{problem}</p>}
        {labels.geometry?.visibility==='occluded'&&<p className="text-amber-300">Hidden / unsure frames are excluded from ball-detector training; they are not absent-ball negatives.</p>}
        <p className="text-slate-400 text-xs">Enter: Save & next · ← / →: change frame. Previous boxes and model predictions are suggestions; inspect each frame. Reuse is limited to adjacent frames in the same burst.</p>
      </div>
      <p className="text-amber-300 text-xs font-mono">{frame.review_metadata?.reasons?.join(' · ')??'Explicit training capture'}</p>
      <ReviewFrameImage enabled={!busy} src={'/api/training-review?clip='+encodeURIComponent(clip)+'&file='+encodeURIComponent(frame.file)} onAnnotation={finish}>
          {(scope==='ball'?['ball'] as Kind[]:['ball','person','table'] as Kind[]).flatMap(k=>labels[k].map((b,i)=><rect key={k+i} x={b[0]*100} y={b[1]*100} width={b[2]*100} height={b[3]*100} fill="none" stroke={k==='ball'?'#fbbf24':k==='person'?'#a78bfa':'#34d399'} strokeWidth=".25"/>))}
          {scope==='all'&&Object.entries(labels.geometry?.landmarks??{}).map(([name,p])=><g key={name}><circle cx={p[0]*100} cy={p[1]*100} r=".65" fill="#34d399"/><text x={p[0]*100+1} y={p[1]*100} fontSize="1.8" fill="white">{name}</text></g>)}
      </ReviewFrameImage>
      <div className="flex flex-wrap gap-3 text-sm">{(scope==='ball'?['ball'] as Kind[]:['ball','person','table'] as Kind[]).map(k=><button key={k} onClick={()=>{setKind(k);setLandmark('');}} className={'px-3 py-2 rounded-lg border '+(kind===k&&!landmark?'border-cyan-400 text-cyan-300':'border-slate-700')}>{k} · {labels[k].length}</button>)}
        <button disabled={busy} onClick={()=>setLabels(l=>({...l,[kind]:l[kind].slice(0,-1)}))}>Undo {kind}</button>
        <button disabled={busy} onClick={()=>setLabels(l=>({...l,[kind]:[]}))}>Clear {kind}</button>
        <button disabled={busy||!frame.review_metadata?.suggestions} onClick={()=>setLabels(l=>scope==='ball'?{...l,ball:frame.review_metadata?.suggestions?.ball??[],geometry:{...(l.geometry??emptyGeometry()),visibility:'unknown'}}:frame.review_metadata?.suggestions??emptyLabels())}>Load model suggestions</button>
      </div>
      {scope==='all'&&<details className="border border-slate-700 rounded-xl p-4 text-sm"><summary>Optional table geometry, blur and event labels</summary><fieldset className="space-y-3 pt-3"><legend className="sr-only">Table geometry and temporal labels</legend>
        <p className="text-slate-400">Near/far refer to the camera; left/right refer to the image. Select a landmark, then click its visible location. Omitted landmarks remain unknown. Return to a box class to draw boxes.</p>
        <select aria-label="Table landmark" value={landmark} onChange={e=>setLandmark(e.target.value as LandmarkName|'')} className="bg-slate-900 p-2"><option value="">Box annotation</option>{landmarkNames.map(n=><option key={n}>{n}</option>)}</select>
        <button className="ml-3 text-amber-300" disabled={!landmark||busy} onClick={()=>setLabels(l=>{const g=l.geometry??emptyGeometry(),points={...g.landmarks};if(landmark)delete points[landmark];return {...l,geometry:{...g,landmarks:points,landmarks_reviewed:false}};})}>Clear selected landmark</button>
        <label className="block"><input type="checkbox" checked={labels.geometry?.landmarks_reviewed??false} onChange={e=>setLabels(l=>({...l,geometry:{...(l.geometry??emptyGeometry()),landmarks_reviewed:e.target.checked}}))}/> I checked the visible landmarks</label>
        <div className="flex flex-wrap gap-4">{(['visibility','blur'] as const).map(field=><label key={field}>{field} <select aria-label={field} className="bg-slate-900 p-2" value={labels.geometry?.[field]??'unknown'} onChange={e=>setLabels(l=>({...l,geometry:{...(l.geometry??emptyGeometry()),[field]:e.target.value}}))}>{(field==='visibility'?['unknown','visible','partial','occluded','absent']:['unknown','none','mild','moderate','severe','extreme']).map(v=><option key={v}>{v}</option>)}</select></label>)}</div>
        <p className="text-slate-400">Events apply at this captured source frame and its recorded elapsed time. A sparse clip may miss the actual contact; leave it unreviewed when uncertain.</p>
        <div className="flex flex-wrap gap-3">{eventNames.map(event=><label key={event}><input type="checkbox" checked={labels.geometry?.events.includes(event)??false} onChange={e=>setLabels(l=>{const g=l.geometry??emptyGeometry();return {...l,geometry:{...g,events:e.target.checked?[...g.events,event]:g.events.filter(x=>x!==event),events_reviewed:false}};})}/> {event}</label>)}</div>
        <label className="block"><input type="checkbox" checked={labels.geometry?.events_reviewed??false} onChange={e=>setLabels(l=>({...l,geometry:{...(l.geometry??emptyGeometry()),events_reviewed:e.target.checked}}))}/> I checked the events at this frame (including no event)</label>
      </fieldset></details>}
      <p className="text-xs text-slate-500">Review does not assign dataset splits or train/deploy a model. Keep complete recording sessions in separate train, validation and test groups using TRAINING_LIVE.md.</p>
    </>}
    {message&&<p role="status" className="text-amber-300">{message}</p>}
  </main>;
}
