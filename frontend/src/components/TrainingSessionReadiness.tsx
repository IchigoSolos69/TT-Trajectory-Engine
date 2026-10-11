'use client';
import {useEffect,useState} from 'react';
import {TrainingDomain,TrainingReadiness,TrainingRegistry,TrainingStage} from '../lib/trainingSessions';

type SessionResponse={registry:TrainingRegistry;revision:string;readiness:TrainingReadiness;clips:Array<{clip:string;original_name:string;original_session_id:string|null}>};
const stageLabels:Record<TrainingStage,string>={existing:'Existing session · training',A:'Session A · training',B:'Session B · validation',C:'Session C · sealed test'};
const domainLabels:Record<TrainingDomain,string>={iphone:'iPhone original recording',pc_camera:'Actual PC camera'};
export function TrainingSessionReadiness({clip,originalSession,onSaved,refreshKey}:{clip?:string;originalSession?:string;onSaved?:()=>void;refreshKey?:string}){
  const [data,setData]=useState<SessionResponse|null>(null),[message,setMessage]=useState(''),[loading,setLoading]=useState(false);
  const [session,setSession]=useState(''),[stage,setStage]=useState<TrainingStage|''>(''),[domain,setDomain]=useState<TrainingDomain|''>(''),[confirmed,setConfirmed]=useState(false);
  async function refresh(){
    setLoading(true);
    try{const response=await fetch('/api/training-sessions',{cache:'no-store'}),value=await response.json();if(!response.ok)throw Error(value.error);setData(value);setMessage('');}
    catch(error){setMessage(String(error));}finally{setLoading(false);}
  }
  useEffect(()=>{void refresh();},[clip,refreshKey]);
  const preservedSession=originalSession??data?.clips.find(value=>value.clip===clip)?.original_session_id??undefined;
  useEffect(()=>{
    const assignment=clip?data?.registry.captures[clip]:undefined;
    setSession(assignment?.session_id??preservedSession??'');setDomain(assignment?.domain??'');setStage(assignment?data?.registry.sessions[assignment.session_id]?.stage??'':'');setConfirmed(false);
  },[clip,preservedSession,data?.registry]);
  async function save(){
    if(!clip||!data||!stage||!domain||!confirmed||loading)return;
    setLoading(true);setMessage('');
    try{
      const response=await fetch('/api/training-sessions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({clip,session_id:session,stage,domain,revision:data.revision,confirm_assignment:confirmed})}),value=await response.json();
      if(!response.ok)throw Error(value.error);
      await refresh();setMessage('Session and device assignment saved separately. Original footage and reviewed labels were preserved.');onSaved?.();
    }catch(error){setMessage(String(error));}finally{setLoading(false);}
  }
  const assignment=clip?data?.registry.captures[clip]:undefined;
  const progress=data?.readiness,training=progress?[progress.counts.train.iphone,progress.counts.train.pc_camera]:[];
  return <section aria-label="Paired training sessions and readiness" className="rounded-xl border border-cyan-500/25 bg-slate-950/40 p-4 space-y-4 text-sm">
    <div className="flex flex-wrap items-center justify-between gap-3"><h3 className="font-semibold text-cyan-200">Paired-camera training plan</h3><button disabled={loading} onClick={()=>void refresh()} className="rounded-lg border border-slate-700 px-3 py-1 text-xs disabled:opacity-40">{loading?'Checking saved labels…':'Refresh collection progress'}</button></div>
    <p className="text-xs leading-relaxed text-slate-300">Record three separate gameplay sessions with both the iPhone and the actual PC camera: <strong>A → training, B → validation, C → sealed test</strong>. Keep simultaneous clips and all repeats from one session together. Session C is reserved for final acceptance; it must never train a model or select thresholds.</p>
    {clip&&<div className="rounded-lg border border-slate-700 p-3 space-y-3" aria-label="Capture session assignment">
      <p className="text-xs text-slate-400">Assign this saved clip explicitly. No session role or recording device is inferred from its filename. Saved assignments are fixed to prevent accidental split leakage.</p>
      <div className="grid gap-3 md:grid-cols-3">
        <label className="text-xs">Whole gameplay session ID<input aria-label="Whole gameplay session ID" value={session} disabled={loading||!!preservedSession||!!assignment} onChange={event=>{setSession(event.target.value);setConfirmed(false);}} placeholder="e.g. venue_20261012_session_A" className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-900 p-2 disabled:opacity-70"/></label>
        <label className="text-xs">Session role<select aria-label="Session role" value={stage} disabled={loading||!!assignment} onChange={event=>{setStage(event.target.value as TrainingStage|'');setConfirmed(false);}} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-900 p-2"><option value="">Unassigned · choose explicitly</option>{Object.entries(stageLabels).map(([value,label])=><option key={value} value={value} disabled={preservedSession==='iphone_same_venue_session_1'&&value!=='existing'}>{label}</option>)}</select></label>
        <label className="text-xs">Recording device<select aria-label="Recording device" value={domain} disabled={loading||!!assignment} onChange={event=>{setDomain(event.target.value as TrainingDomain|'');setConfirmed(false);}} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-900 p-2"><option value="">Unknown · choose explicitly</option>{Object.entries(domainLabels).map(([value,label])=><option key={value} value={value}>{label}</option>)}</select></label>
      </div>
      {preservedSession&&<p className="text-xs text-amber-200">Original session ID preserved: {preservedSession}.{preservedSession==='iphone_same_venue_session_1'?' All these recordings belong to the same earlier development session; choose “Existing session · training”.':''}</p>}
      {!assignment?<><label className="flex items-start gap-2 text-xs text-amber-100"><input aria-label="Confirm whole-session role and device" type="checkbox" checked={confirmed} disabled={loading||!session||!stage||!domain} onChange={event=>setConfirmed(event.target.checked)}/>I checked the original recording device and whole session identity. Every clip from this gameplay session will keep this one dataset role.</label><button disabled={loading||!session||!stage||!domain||!confirmed} onClick={()=>void save()} className="rounded-lg bg-cyan-400 px-3 py-2 text-xs font-bold text-slate-950 disabled:opacity-40">Save session & device assignment</button></>:<p className="text-xs text-emerald-300">Saved: {stageLabels[data!.registry.sessions[assignment.session_id].stage]} · {domainLabels[assignment.domain]}{data!.registry.sessions[assignment.session_id].sealed?' · SEALED FOR FINAL ACCEPTANCE':''}</p>}
    </div>}
    {progress&&<>
      <div className="grid gap-2 md:grid-cols-3" aria-label="First training cycle collection progress">{([
        ['Reviewed visible ball boxes',training.reduce((sum,value)=>sum+value.visible_boxes,0),2000],['Verified absent saved frames',training.reduce((sum,value)=>sum+value.absent_frames,0),1000],['Positive three-frame windows',training.reduce((sum,value)=>sum+value.positive_windows,0),500]
      ] as Array<[string,number,number]>).map(([label,count,target])=><div key={label} className="rounded-lg border border-slate-700 bg-slate-900/50 p-3"><p className="text-xs text-slate-400">{label}</p><p className="mt-1 font-mono text-cyan-200">{count.toLocaleString()} / {target.toLocaleString()}</p><progress aria-label={label} value={Math.min(count,target)} max={target} className="mt-2 h-1.5 w-full accent-cyan-400"/></div>)}</div>
      <div className="overflow-x-auto"><table className="w-full text-left text-xs">
        <caption className="mb-2 text-left text-slate-400">Explicitly assigned camera recordings only. Public match training is audited separately.</caption>
        <thead className="text-slate-400"><tr><th className="p-2">Role / device</th><th className="p-2">Visible frames</th><th className="p-2">Absent frames</th><th className="p-2">Precise center targets</th><th className="p-2">Reviewed rally IDs</th><th className="p-2">Verified no-ball coverage</th></tr></thead>
        <tbody>{(['train','val','test'] as const).flatMap(role=>(['iphone','pc_camera'] as const).map(device=>{
          if(role==='test')return <tr key={role+device} className="border-t border-slate-800"><td className="p-2">sealed test · {device==='iphone'?'iPhone':'PC camera'}</td><td colSpan={5} className="p-2 text-amber-200">Sealed · not audited before final acceptance</td></tr>;
          const value=progress.counts[role][device];
          return <tr key={role+device} className="border-t border-slate-800"><td className="p-2">{role==='val'?'validation':'training'} · {device==='iphone'?'iPhone':'PC camera'}</td><td className="p-2 font-mono">{value.visible_frames}{role==='val'?' / 1,000':''}</td><td className="p-2 font-mono">{value.absent_frames}</td><td className="p-2 font-mono">{value.precise_points}</td><td className="p-2 text-amber-200">{value.rallies??'Unknown'}{role==='val'?' / 10':''}</td><td className="p-2 text-amber-200">{value.verified_absence_seconds===null?'Unknown':value.verified_absence_seconds.toFixed(2)+' s'} / 120 s</td></tr>;
        }))}</tbody>
      </table></div>
      <p className="text-xs text-amber-200">Awaiting audited training data. {progress.unassigned_captures} clip(s) remain unassigned: {progress.unassigned.visible_boxes} visible boxes, {progress.unassigned.absent_frames} absent saved frames and {progress.unassigned.positive_windows} positive windows. They are excluded from split progress. {progress.conflicting_source_frames} conflicting source frame(s), {progress.conflicting_original_sources} conflicting original source(s) and {progress.duplicate_images} duplicate image(s) are excluded.</p>
      <p className="text-xs text-slate-400">For validation and test, each device needs 1,000 reviewed visible frames across at least 10 complete rallies, plus 2 minutes of reviewed original no-ball footage. Sampled absent images do not prove a continuous absence interval. Positive windows count new session A bursts only, with explicit continuity review and a reviewed center/streak on the latest of three consecutive original images.</p>
      <details className="text-xs text-slate-400"><summary className="cursor-pointer text-cyan-200">How to supply the 2-minute absence evidence</summary><ol className="mt-2 list-decimal pl-4 space-y-1"><li>Keep the original continuous iPhone and PC-camera files with the ball physically removed. Saved five-frame bursts are insufficient.</li><li>Assign the imported original recording to its whole session and device. In Training review, open Full original no-ball interval and prepare only the exact range you intend to review.</li><li>Inspect every prepared frame of that interval, then explicitly confirm the entire interval contains no ball, including no hidden or uncertain ball. Complete decoded timestamps and immutable source/image hashes are required before its duration is credited.</li><li>The separate No-ball interval control verifies only sampled saved images. Preparation requires a local Python/OpenCV training runtime; native analysis and annotation remain available without it.</li></ol></details>
      <details className="text-xs text-slate-400"><summary className="cursor-pointer text-amber-200">Remaining data checks · no accuracy claim yet</summary><ul className="mt-2 list-disc pl-4 space-y-1">{progress.limitations.map(value=><li key={value}>{value}</li>)}<li>Review 10% of labels a second time; resolve center disagreements over 2 px. Labels and all partitions need a frozen audit before training.</li></ul></details>
    </>}
    {message&&<p role="status" className="text-xs text-amber-200">{message}</p>}
  </section>;
}
