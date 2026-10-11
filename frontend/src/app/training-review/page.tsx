'use client';
import {useEffect,useState} from 'react';
import {landmarkNames,eventNames,emptyGeometry,LandmarkName,createBlurStreak} from '../../lib/reviewGeometry';
import {ReviewFrameImage} from '../../components/ReviewFrameImage';
import {TrainingSessionReadiness} from '../../components/TrainingSessionReadiness';
import {BurstContinuityReview} from '../../components/BurstContinuityReview';
import {FullAbsenceIntervalReview} from '../../components/FullAbsenceIntervalReview';
import {Kind,Box,Labels,Annotation,Frame,emptyLabels,classReviewed,ballReviewProblem,previousBallSuggestion,nextBallDraft,nextBallReviewIndex,validDraft} from '../../lib/reviewAnnotations';
type Capture={source?:string;original_name?:string;session_id?:string;source_fps_average?:number;frame_count_discrepancy?:number;capture_note?:string;full_source_decode_counted?:boolean;source_frames?:number};
type AbsencePreview={preview_token:string;affected_frames:number;frames:Frame[];conflicts:Array<{file:string;source_frame:number}>;session_id:string};
const draftStorageKey='tt-training-review-drafts-v1';
export default function TrainingReview(){
  const [clips,setClips]=useState<string[]>([]),[clip,setClip]=useState(''),[frames,setFrames]=useState<Frame[]>([]),[index,setIndex]=useState(0);
  const [drafts,setDrafts]=useState<Record<string,Labels>>({}),[annotations,setAnnotations]=useState<Record<string,Annotation>>({});
  const [revision,setRevision]=useState(''),[message,setMessage]=useState(''),[kind,setKind]=useState<Kind>('ball'),[busy,setBusy]=useState(false);
  const [capture,setCapture]=useState<Capture|null>(null);
  const [readinessVersion,setReadinessVersion]=useState(0);
  const [sealedTest,setSealedTest]=useState(false);
  const [clipNames,setClipNames]=useState<Record<string,string>>({});
  const [landmark,setLandmark]=useState<LandmarkName|''>('');
  const [annotationMode,setAnnotationMode]=useState<'box'|'streak'>('box');
  const [scope,setScope]=useState<'ball'|'all'>('ball'),[draftsRestored,setDraftsRestored]=useState(false);
  const [absenceStart,setAbsenceStart]=useState(1),[absenceEnd,setAbsenceEnd]=useState(1),[absencePreview,setAbsencePreview]=useState<AbsencePreview|null>(null),[absenceConfirmed,setAbsenceConfirmed]=useState(false);
  const frame=frames[index];
  const key=clip+'/'+(frame?.file??''),labels=drafts[key]??annotations[frame?.file??'']??emptyLabels();
  const problem=ballReviewProblem(labels),previousBall=previousBallSuggestion(frames,index,annotations);
  const nextSuggestion=nextBallDraft(frames,index,labels,annotations,drafts,clip);
  const burst=frame?.burst_id===undefined?[]:frames.filter(value=>value.burst_id===frame.burst_id);
  const clipHasDrafts=Object.keys(drafts).some(draft=>draft.startsWith(clip+'/'));
  useEffect(()=>{setAbsencePreview(null);setAbsenceConfirmed(false);},[clip,revision,absenceStart,absenceEnd]);
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
      setClip(value);setFrames(d.frames);setAnnotations(d.annotations);setRevision(d.revision);setIndex(0);setCapture(d.capture??null);setSealedTest(d.sealed_test===true);
      setAbsenceStart(d.frames[0]?.source_frame??1);setAbsenceEnd(d.frames.at(-1)?.source_frame??1);setAbsencePreview(null);setAbsenceConfirmed(false);
    }catch(e){setMessage(String(e));}finally{setBusy(false);}
  }
  function finish(a:[number,number],b:[number,number],size:{width:number;height:number}){
    if(busy)return;
    if(landmark){setLabels(previous=>({...previous,geometry:{...(previous.geometry??emptyGeometry()),landmarks:{...previous.geometry?.landmarks,[landmark]:b},landmarks_reviewed:false}}));return;}
    if(annotationMode==='streak'){
      setLabels(previous=>({...previous,geometry:{...(previous.geometry??emptyGeometry()),
        visibility:previous.geometry?.visibility==='partial'?'partial':'visible',blur_streak:createBlurStreak(a,b,size)}}));
      return;
    }
    let box:Box=[Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.abs(a[0]-b[0]),Math.abs(a[1]-b[1])];
    if(kind==='ball'&&box[2]*size.width<2&&box[3]*size.height<2){const w=labels.ball[0]?.[2]??previousBall?.[2]??Math.min(1,12/size.width),h=labels.ball[0]?.[3]??previousBall?.[3]??Math.min(1,12/size.height);box=[Math.max(0,Math.min(1-w,b[0]-w/2)),Math.max(0,Math.min(1-h,b[1]-h/2)),w,h];}
    if(box[2]>0&&box[3]>0)setLabels(previous=>({...previous,[kind]:scope==='ball'?[box]:[...previous[kind],box],geometry:kind==='ball'?{...(previous.geometry??emptyGeometry()),visibility:'visible'}:previous.geometry}));
  }
  function reuseBall(){if(previousBall){setKind('ball');setLandmark('');setAnnotationMode('box');setLabels(l=>({...l,ball:[previousBall],geometry:{...(l.geometry??emptyGeometry()),visibility:'visible'}}));setMessage('Previous box copied as a draft. Check and move it to the current ball before saving.');}}
  function suggestNextBall(){
    if(!nextSuggestion||busy)return;
    setDrafts(previous=>({...previous,[nextSuggestion.key]:nextSuggestion.labels}));setIndex(nextSuggestion.index);setKind('ball');setLandmark('');setAnnotationMode('box');
    setMessage('Adjacent box copied as an unconfirmed draft. Inspect and correct this image, choose its visibility and save explicitly. Existing reviewed frames and drafts were preserved.');
  }
  async function save(reviewed:boolean,advance=false){
    if(!frame||busy)return;setBusy(true);setMessage('');
    try{const r=await fetch('/api/training-review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({clip,file:frame.file,labels,reviewed,revision,scope})}),d=await r.json();if(!r.ok)throw Error(d.error);
      setRevision(d.revision);setAnnotations(a=>({...a,[frame.file]:d.annotation}));setDrafts(a=>{const next={...a};delete next[key];return next;});setMessage(reviewed?(scope==='ball'?'Ball label saved. People/table review status preserved.':'Reviewed labels saved locally.'):'Selected labels marked unreviewed; excluded from their training dataset.');
      if(advance&&index<frames.length-1)setIndex(index+1);
    }catch(e){setMessage(String(e));}finally{setBusy(false);}
  }
  async function reviewAbsence(confirm=false){
    if(busy||clipHasDrafts||!draftsRestored||!clip||(confirm&&(!absencePreview||!absenceConfirmed)))return;
    setBusy(true);setMessage('');
    try{
      const response=await fetch('/api/training-review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({clip,revision,start_source_frame:absenceStart,end_source_frame:absenceEnd,action:confirm?'confirm_ball_absence_interval':'preview_ball_absence_interval',preview_token:confirm?absencePreview?.preview_token:undefined,confirm_absence:confirm?absenceConfirmed:undefined})});
      const result=await response.json();if(!response.ok){setAbsencePreview(null);setAbsenceConfirmed(false);throw Error(result.error);}
      if(confirm){setRevision(result.revision);setAnnotations(result.annotations);setAbsencePreview(null);setAbsenceConfirmed(false);setMessage(`Ball absence saved for ${result.affected_frames} sampled images. People/table review status preserved.`);}
      else{setAbsencePreview(result);setAbsenceConfirmed(false);}
    }catch(error){setMessage(String(error));}finally{setBusy(false);}
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
    <p className="text-slate-400 text-sm">Ball-only review: click the ball for a box, or select Blur streak and drag from one visible end to the other, then Save & next. You do not need to label people, table corners or events. Streak labels preserve existing boxes and prepare temporal/blur training; box-detector training still needs boxes.</p>
    <div className="flex flex-wrap gap-3 text-sm"><button disabled={busy} aria-pressed={scope==='ball'} onClick={()=>{setScope('ball');setKind('ball');setLandmark('');setAnnotationMode('box');}} className="border border-cyan-500/40 rounded-lg px-3 py-2">Ball only · faster</button><button disabled={busy} aria-pressed={scope==='all'} onClick={()=>setScope('all')} className="border border-slate-700 rounded-lg px-3 py-2">Full scene & advanced</button></div>
    <select aria-label="Capture clip" value={clip} disabled={busy} onChange={e=>load(e.target.value)} className="bg-slate-900 border border-slate-700 rounded-lg p-2"><option value="">Select a captured clip</option>{clips.map(c=><option key={c} value={c}>{clipNames[c]??c}</option>)}</select>
    {!clips.length&&<p>No local training captures yet. Use the live-camera recording controls or import original camera recordings first.</p>}
    <TrainingSessionReadiness key={clip} clip={clip||undefined} originalSession={capture?.session_id} refreshKey={revision+'-'+readinessVersion} onSaved={()=>{if(clip)void load(clip);}}/>
    {sealedTest&&<p role="status" className="rounded-xl border border-amber-500/40 bg-amber-500/5 p-3 text-sm text-amber-200">Sealed test: manual annotation only. Model suggestions are disabled, and its labels/frames are excluded from automated collection readiness and model development. Final evaluation waits until models and thresholds are frozen.</p>}
    {capture?.source==='recorded_camera'&&<div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-4 text-sm space-y-2">
      <p className="text-cyan-300">{capture.original_name} · {capture.source_fps_average?.toFixed(2)} average FPS · session {capture.session_id}</p>
      {capture.capture_note&&<p className="text-amber-200">Review focus: {capture.capture_note}</p>}
      <p className="text-amber-200">Source recording imported as unreviewed consecutive-frame bursts. Keep every clip from this session in the same dataset split. These frames do not become training labels until you explicitly review and save them. Before temporal training, check bursts for cuts, repeated frames and capture interruptions.</p>
      {!!capture.frame_count_discrepancy&&<p className="text-amber-200">Reported and decoded frame counts differ by {capture.frame_count_discrepancy}. Imported frames use sequential decoding and source timestamps; recording playback timing still needs separate verification.</p>}
    </div>}
    {!!Object.keys(drafts).length&&<p className="text-amber-300 text-sm">{Object.keys(drafts).length} frame(s) have unconfirmed drafts stored locally. Drafts stay excluded from training until reviewed and saved.</p>}
    {clip&&capture?.source==='recorded_camera'&&<FullAbsenceIntervalReview key={clip+'-'+sealedTest} clip={clip} sealed={sealedTest} disabled={busy||clipHasDrafts||!draftsRestored} sourceFrames={capture.source_frames} onSaved={()=>setReadinessVersion(value=>value+1)}/>}
    {capture?.source==='recorded_camera'&&capture.full_source_decode_counted&&frames.length>0&&<details className="rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-4 text-sm space-y-3">
      <summary className="cursor-pointer font-semibold text-emerald-300">No-ball interval · review saved images together</summary>
      <p>Use this for a recording interval you know contains no ball. Only saved images in the interval will be labelled; unsampled footage is not labelled. Hidden or uncertain balls need individual review. People and table labels keep their current status.</p>
      <div className="flex flex-wrap items-center gap-3">
        <label>First source frame <input aria-label="No-ball first source frame" type="number" min={1} max={capture.source_frames} value={absenceStart} disabled={busy} onChange={event=>setAbsenceStart(Number(event.target.value))} className="w-28 rounded bg-slate-900 border border-slate-700 p-2"/></label>
        <label>Last source frame <input aria-label="No-ball last source frame" type="number" min={absenceStart} max={capture.source_frames} value={absenceEnd} disabled={busy} onChange={event=>setAbsenceEnd(Number(event.target.value))} className="w-28 rounded bg-slate-900 border border-slate-700 p-2"/></label>
        <button disabled={busy||clipHasDrafts||!draftsRestored} onClick={()=>reviewAbsence()} className="rounded-lg border border-emerald-500/40 px-3 py-2">Preview no-ball interval</button>
      </div>
      {clipHasDrafts&&<p role="alert" className="text-amber-300">Save or resolve this clip’s pending drafts before bulk review. Your drafts will not be overwritten.</p>}
      {absencePreview&&<div className="space-y-3">
        <p>{absencePreview.affected_frames} saved image(s) selected · session {absencePreview.session_id}. Review the selected frames before confirming. Keep all clips from this session in one dataset split.</p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">{absencePreview.frames.filter((_,i,all)=>i===0||i===all.length-1||i%Math.max(1,Math.ceil(all.length/10))===0).slice(0,12).map(selected=><button key={selected.file} disabled={busy} onClick={()=>setIndex(frames.findIndex(value=>value.file===selected.file))} className="rounded border border-slate-700 p-1 text-left"><img alt={'No-ball preview source frame '+selected.source_frame} src={'/api/training-review?clip='+encodeURIComponent(clip)+'&file='+encodeURIComponent(selected.file)} className="w-full"/><span className="font-mono text-xs">Source frame {selected.source_frame}</span></button>)}</div>
        {absencePreview.conflicts.length>0?<p role="alert" className="text-amber-300">{absencePreview.conflicts.length} frame(s) contain existing visible or hidden ball evidence. Resolve them individually before bulk absence review: {absencePreview.conflicts.slice(0,10).map(conflict=>conflict.source_frame).join(', ')}.</p>:<>
          <label className="flex items-start gap-2"><input aria-label="Confirm no ball in selected images" type="checkbox" checked={absenceConfirmed} disabled={busy||clipHasDrafts} onChange={event=>setAbsenceConfirmed(event.target.checked)}/>I verified no ball is present in the selected saved images; uncertain or hidden-ball images must remain unreviewed.</label>
          <button disabled={busy||clipHasDrafts||!absenceConfirmed} onClick={()=>reviewAbsence(true)} className="rounded-lg bg-emerald-400 text-slate-950 font-bold px-3 py-2">Confirm ball absence for selected images</button>
        </>}
      </div>}
    </details>}
    {frame&&<>
      <div className="flex gap-3 flex-wrap items-center font-mono text-xs"><button disabled={busy||index===0} onClick={()=>setIndex(index-1)}>Previous</button><span>{index+1} / {frames.length} · source frame {frame.source_frame} · {classReviewed(annotations[frame.file],'ball')?'BALL REVIEWED':'BALL UNREVIEWED'}</span><button disabled={busy||index===frames.length-1} onClick={()=>setIndex(index+1)}>Next</button><button disabled={busy} onClick={()=>{const next=nextBallReviewIndex(frames,index,annotations);if(next>=0)setIndex(next);else setMessage('No remaining ball frames need review in this capture.');}}>Next needing review</button></div>
      {burst.length>1&&<details open className="rounded-xl border border-slate-700 bg-slate-900/30 p-3 text-xs" aria-label="Consecutive burst review"><summary className="cursor-pointer text-cyan-200">Burst {frame.burst_id} · {burst.length} saved frames · {burst.filter(value=>classReviewed(annotations[value.file],'ball')).length} ball labels saved</summary><p className="mt-2 text-slate-400">Inspect nearby images together. Each visible center, absent label or correction still needs an explicit save; missing images and hidden balls are never interpolated into ground truth.</p><div className="mt-3 grid grid-cols-3 gap-2 sm:grid-cols-5">{burst.map(value=><button key={value.file} disabled={busy} aria-label={'Review burst source frame '+value.source_frame} onClick={()=>setIndex(frames.findIndex(candidate=>candidate.file===value.file))} className={'rounded-lg border p-1 text-left '+(value.file===frame.file?'border-cyan-400':'border-slate-700')}><img alt={'Burst source frame '+value.source_frame} src={'/api/training-review?clip='+encodeURIComponent(clip)+'&file='+encodeURIComponent(value.file)} className="w-full rounded"/><span className="mt-1 block font-mono">{value.source_frame} · {drafts[clip+'/'+value.file]?'draft':classReviewed(annotations[value.file],'ball')?'saved':'unreviewed'}</span></button>)}</div><BurstContinuityReview key={clip+'/'+frame.burst_id} clip={clip} burst={burst} onSaved={()=>setReadinessVersion(value=>value+1)}/></details>}
      <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/5 p-3 space-y-3 text-sm">
        <div className="flex flex-wrap gap-3"><button disabled={busy||!!problem} onClick={()=>save(true,true)} className="bg-emerald-400 text-slate-950 font-bold rounded-lg px-4 py-2">Save & next</button><button disabled={busy||!!problem} onClick={()=>save(true)} className="border border-slate-700 rounded-lg px-3 py-2">Save this frame</button><button disabled={busy} onClick={()=>save(false,true)} className="border border-slate-700 rounded-lg px-3 py-2">Skip / keep unreviewed</button><button disabled={busy||!previousBall} onClick={reuseBall} className="border border-slate-700 rounded-lg px-3 py-2">Reuse previous ball box</button></div>
        <button disabled={busy||!nextSuggestion} onClick={suggestNextBall} className="rounded-lg border border-cyan-500/40 px-3 py-2 disabled:opacity-40">Suggest box on next burst frame</button>
        <div className="flex flex-wrap gap-3">{(['visible','partial','absent','occluded'] as const).map((visibility,i)=><button key={visibility} disabled={busy} aria-pressed={labels.geometry?.visibility===visibility} onClick={()=>setLabels(l=>({...l,ball:visibility==='absent'||visibility==='occluded'?[]:l.ball,geometry:{...(l.geometry??emptyGeometry()),visibility,blur_streak:visibility==='absent'||visibility==='occluded'?null:l.geometry?.blur_streak??null}}))} className="border border-slate-700 rounded-lg px-3 py-2">{['Visible','Partly visible','Not in frame','Hidden / unsure'][i]}</button>)}</div>
        <div className="flex flex-wrap gap-3"><button disabled={busy} aria-pressed={annotationMode==='box'&&!landmark} onClick={()=>{setAnnotationMode('box');setKind('ball');setLandmark('');}} className="border border-slate-700 rounded-lg px-3 py-2">Ball box</button><button disabled={busy} aria-pressed={annotationMode==='streak'&&!landmark} onClick={()=>{setAnnotationMode('streak');setKind('ball');setLandmark('');}} className="border border-cyan-500/40 rounded-lg px-3 py-2">Blur streak · drag endpoints</button><button disabled={busy||!labels.geometry?.blur_streak} onClick={()=>setLabels(l=>({...l,geometry:{...(l.geometry??emptyGeometry()),blur_streak:null}}))} className="border border-slate-700 rounded-lg px-3 py-2">Clear streak</button></div>
        {annotationMode==='streak'&&!landmark&&<p className="text-cyan-200 text-xs">Drag only the visible streak’s two ends. For a sharp ball, click its center. Do not extend the label through a player or guess a hidden ball. Save explicitly confirms this frame; no model suggestion becomes a reviewed label automatically.</p>}
        <p className="text-xs text-slate-400">For precise temporal targets, use Blur streak mode: click the center of a sharp ball or drag the two ends of its visible streak. A box is still required separately for YOLO training. Bright lights and white walls are not balls; preserve those false detections as explicitly reviewed negatives only when the real ball is absent.</p>
        <div className="flex flex-wrap gap-3 text-xs" aria-label="Reviewed ball difficulty tags">{(['bright_background','small_ball','motion_blur','partial_occlusion'] as const).map(tag=><label key={tag} className="flex gap-2"><input type="checkbox" disabled={busy} checked={labels.geometry?.quality_tags?.includes(tag)??false} onChange={event=>setLabels(value=>{const geometry=value.geometry??emptyGeometry(),tags=geometry.quality_tags??[];return {...value,geometry:{...geometry,quality_tags:event.target.checked?[...tags,tag]:tags.filter(item=>item!==tag)}};})}/>{tag.replaceAll('_',' ')}</label>)}</div>
        <details className="rounded-lg border border-slate-700 p-3 text-xs"><summary className="cursor-pointer text-cyan-200">Optional complete-rally identity</summary><p className="mt-2 text-slate-400">Watch the complete rally in the original recording, including its start and end. Give all its sampled frames one source-specific ID (for example rally_01). A guessed strike or a sparse frame does not prove a complete rally.</p><label className="mt-2 block">Rally ID<input aria-label="Reviewed complete rally ID" value={labels.geometry?.rally_id??''} disabled={busy} onChange={event=>setLabels(value=>({...value,geometry:{...(value.geometry??emptyGeometry()),rally_id:event.target.value||null,rally_reviewed:false}}))} placeholder="rally_01" className="ml-2 rounded border border-slate-700 bg-slate-900 p-2"/></label><label className="mt-2 flex items-start gap-2"><input aria-label="Confirm complete rally reviewed in original footage" type="checkbox" disabled={busy||!labels.geometry?.rally_id} checked={labels.geometry?.rally_reviewed??false} onChange={event=>setLabels(value=>({...value,geometry:{...(value.geometry??emptyGeometry()),rally_reviewed:event.target.checked}}))}/>I reviewed the full original rally and verified that this image belongs to this ID. Save this frame explicitly to confirm it.</label></details>
        {labels.geometry?.blur_streak&&<p className="font-mono text-xs text-cyan-200">Streak center {(labels.geometry.blur_streak.center[0]*labels.geometry.blur_streak.source_width).toFixed(1)}, {(labels.geometry.blur_streak.center[1]*labels.geometry.blur_streak.source_height).toFixed(1)} px · half-length {labels.geometry.blur_streak.half_length_px.toFixed(1)} px · {labels.geometry.blur_streak.reviewed?'saved reviewed label':'unconfirmed draft'}. Axis is undirected; this is not spin or measured speed.</p>}
        {problem&&<p role="alert" className="text-amber-300">{problem}</p>}
        {labels.geometry?.visibility==='occluded'&&<p className="text-amber-300">Hidden / unsure frames are excluded from ball-detector training; they are not absent-ball negatives.</p>}
        <p className="text-slate-400 text-xs">Enter: Save & next · ← / →: change frame. Previous boxes and model predictions are suggestions; inspect each frame. Reuse is limited to adjacent frames in the same burst.</p>
      </div>
      <p className="text-amber-300 text-xs font-mono">{frame.review_metadata?.reasons?.join(' · ')??'Explicit training capture'}</p>
      <ReviewFrameImage enabled={!busy} src={'/api/training-review?clip='+encodeURIComponent(clip)+'&file='+encodeURIComponent(frame.file)} onAnnotation={finish}>
          {(scope==='ball'?['ball'] as Kind[]:['ball','person','table'] as Kind[]).flatMap(k=>labels[k].map((b,i)=><rect key={k+i} x={b[0]*100} y={b[1]*100} width={b[2]*100} height={b[3]*100} fill="none" stroke={k==='ball'?'#fbbf24':k==='person'?'#a78bfa':'#34d399'} strokeWidth=".25"/>))}
          {scope==='all'&&Object.entries(labels.geometry?.landmarks??{}).map(([name,p])=><g key={name}><circle cx={p[0]*100} cy={p[1]*100} r=".65" fill="#34d399"/><text x={p[0]*100+1} y={p[1]*100} fontSize="1.8" fill="white">{name}</text></g>)}
          {labels.geometry?.blur_streak&&<g><line x1={labels.geometry.blur_streak.endpoints[0][0]*100} y1={labels.geometry.blur_streak.endpoints[0][1]*100} x2={labels.geometry.blur_streak.endpoints[1][0]*100} y2={labels.geometry.blur_streak.endpoints[1][1]*100} stroke="#22d3ee" strokeWidth=".3"/>{labels.geometry.blur_streak.endpoints.map((p,i)=><circle key={i} cx={p[0]*100} cy={p[1]*100} r=".4" fill="#22d3ee"/>)}<circle cx={labels.geometry.blur_streak.center[0]*100} cy={labels.geometry.blur_streak.center[1]*100} r=".45" fill="#fbbf24"/></g>}
      </ReviewFrameImage>
      <div className="flex flex-wrap gap-3 text-sm">{(scope==='ball'?['ball'] as Kind[]:['ball','person','table'] as Kind[]).map(k=><button key={k} onClick={()=>{setKind(k);setLandmark('');setAnnotationMode('box');}} className={'px-3 py-2 rounded-lg border '+(kind===k&&!landmark&&annotationMode==='box'?'border-cyan-400 text-cyan-300':'border-slate-700')}>{k} · {labels[k].length}</button>)}
        <button disabled={busy} onClick={()=>setLabels(l=>({...l,[kind]:l[kind].slice(0,-1)}))}>Undo {kind}</button>
        <button disabled={busy} onClick={()=>setLabels(l=>({...l,[kind]:[]}))}>Clear {kind}</button>
        <button disabled={busy||sealedTest||!frame.review_metadata?.suggestions} onClick={()=>setLabels(l=>scope==='ball'?{...l,ball:frame.review_metadata?.suggestions?.ball??[],geometry:{...(l.geometry??emptyGeometry()),visibility:'unknown'}}:frame.review_metadata?.suggestions??emptyLabels())}>Load model suggestions</button>
      </div>
      {scope==='all'&&<details className="border border-slate-700 rounded-xl p-4 text-sm"><summary>Optional table geometry, blur and event labels</summary><fieldset className="space-y-3 pt-3"><legend className="sr-only">Table geometry and temporal labels</legend>
        <p className="text-slate-400">Near/far refer to the camera; left/right refer to the image. Select a landmark, then click its visible location. Omitted landmarks remain unknown. Return to a box class to draw boxes.</p>
        <select aria-label="Table landmark" value={landmark} onChange={e=>{setLandmark(e.target.value as LandmarkName|'');setAnnotationMode('box');}} className="bg-slate-900 p-2"><option value="">Box annotation</option>{landmarkNames.map(n=><option key={n}>{n}</option>)}</select>
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
