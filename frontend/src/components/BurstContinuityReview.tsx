'use client';
import {useEffect,useState} from 'react';
import {ContinuityAudit} from '../lib/trainingSessions';
import {Frame} from '../lib/reviewAnnotations';
type Proof={index_sha256:string;source_id:string;continuity:ContinuityAudit|null;continuity_revision:string};
export function BurstContinuityReview({clip,burst,onSaved}:{clip:string;burst:Frame[];onSaved:()=>void}){
  const [proof,setProof]=useState<Proof|null>(null),[confirmed,setConfirmed]=useState(false),[busy,setBusy]=useState(false),[message,setMessage]=useState('');
  const first=burst[0],last=burst.at(-1),complete=!!first&&burst.length>=3&&burst.length===first.burst_length&&burst.every((frame,i)=>frame.burst_index===i&&frame.burst_id===first.burst_id&&(i===0||frame.source_frame===burst[i-1].source_frame+1));
  useEffect(()=>{
    let active=true;setProof(null);setConfirmed(false);setMessage('');
    fetch('/api/training-sessions?clip='+encodeURIComponent(clip),{cache:'no-store'}).then(async response=>{const value=await response.json();if(!response.ok)throw Error(value.error);if(active)setProof(value);}).catch(error=>{if(active)setMessage(String(error));});
    return()=>{active=false;};
  },[clip,first?.burst_id]);
  const already=!!proof&&!!first&&!!last&&proof.continuity?.source_frame_ranges.some(([start,end])=>start<=first.source_frame&&last.source_frame<=end);
  async function save(){
    if(!proof||!first||!last||!complete||!confirmed||busy)return;setBusy(true);setMessage('');
    try{
      const response=await fetch('/api/training-sessions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'confirm_burst_continuity',clip,burst_id:first.burst_id,start_source_frame:first.source_frame,end_source_frame:last.source_frame,index_sha256:proof.index_sha256,source_id:proof.source_id,continuity_revision:proof.continuity_revision,confirm_source_continuity:true,confirm_capture_complete:true})}),value=await response.json();
      if(!response.ok)throw Error(value.error);
      setProof({...proof,continuity:value.continuity,continuity_revision:value.continuity_revision});setConfirmed(false);setMessage('Viewed burst continuity saved against this source and exact index. Labels were preserved; this confirms neither ball absence nor a complete rally.');onSaved();
    }catch(error){setMessage(String(error));}finally{setBusy(false);}
  }
  return <div aria-label="Review original burst continuity" className="mt-3 rounded-lg border border-cyan-500/25 p-3 space-y-2 text-xs">
    <p className="font-semibold text-cyan-200">Confirm source continuity for this burst</p>
    <p className="text-slate-400">Frame numbers alone do not establish continuity. After stopping capture, inspect every image above and check the original source for cuts, duplicated images or interruptions. This records only the selected burst; later source/index changes invalidate the audit.</p>
    {!complete?<p className="text-amber-200">This saved burst is incomplete or not consecutive. Exclude it from temporal training and capture/import a complete burst.</p>:already?<p className="text-emerald-300">Confirmed source frames {first.source_frame}–{last!.source_frame} · index {proof!.index_sha256.slice(0,12)}…</p>:<><label className="flex items-start gap-2"><input aria-label="Confirm viewed completed burst continuity" type="checkbox" checked={confirmed} disabled={busy||!proof} onChange={event=>setConfirmed(event.target.checked)}/>I stopped/finished this capture and reviewed these original consecutive images: no cut, duplicated frame or interruption within this burst.</label><button disabled={busy||!proof||!confirmed} onClick={()=>void save()} className="rounded-lg border border-cyan-500/50 px-3 py-2 disabled:opacity-40">Save viewed burst continuity</button></>}
    {message&&<p role="status" className="text-amber-200">{message}</p>}
  </div>;
}
