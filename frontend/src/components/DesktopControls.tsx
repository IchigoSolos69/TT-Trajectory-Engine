'use client';
import {useState} from 'react';
import {useDesktop} from '../hooks/useDesktop';
export function DesktopControls(){
  const {context}=useDesktop();
  const [error,setError]=useState<string|null>(null);
  if(!context)return null;
  const run=async(action:()=>Promise<unknown>)=>{setError(null);try{await action();}catch(e){setError(e instanceof Error?e.message:String(e));}};
  return <div className="border-b border-slate-800 bg-slate-900/80 px-6 py-3 text-xs" aria-label="Desktop session controls">
    <div className="flex flex-wrap items-center gap-3 max-w-7xl mx-auto">
      <span className="font-mono text-emerald-300 mr-auto">LOCAL APP · {context.mode==='live'?'LIVE CAMERA':'RECORDING'}</span>
      <button disabled={context.phase==='starting'} onClick={()=>run(()=>window.ttDesktop!.openRecording())} className="rounded-lg border border-cyan-700 px-3 py-2 text-cyan-300 disabled:opacity-50">Open recording…</button>
      <button disabled={context.phase==='starting'||context.mode==='live'} onClick={()=>run(()=>window.ttDesktop!.startLive())} className="rounded-lg border border-slate-700 px-3 py-2 disabled:opacity-50">Live camera</button>
      <button onClick={()=>run(()=>window.ttDesktop!.showDataDirectory())} className="rounded-lg border border-slate-700 px-3 py-2">Local data</button>
      <button disabled={context.phase==='starting'} onClick={()=>run(()=>window.ttDesktop!.endSession())} className="rounded-lg border border-slate-700 px-3 py-2 disabled:opacity-50">End session</button>
    </div>
    {(error||context.error||context.phase==='starting')&&<p role="status" className="max-w-7xl mx-auto mt-2 text-amber-300">{error||context.error||'Changing local session…'}</p>}
  </div>;
}
