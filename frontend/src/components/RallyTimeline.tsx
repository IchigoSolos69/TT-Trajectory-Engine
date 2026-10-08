'use client';
import type {IntelligenceTelemetry} from '../types/telemetry';
export function RallyTimeline({data,canSeek,sendCommand}:{data?:IntelligenceTelemetry;canSeek:boolean;sendCommand:(command:string)=>void}){
  const graph=data?.rally_graph;if(!graph)return null;
  return <section aria-label="Rally event graph" className="rounded-xl border border-slate-800 bg-slate-950 p-4 space-y-3">
    <div className="flex justify-between gap-4"><h2 className="font-bold text-cyan-300">Rally event timeline</h2><span className="text-xs text-amber-300">Experimental event rules</span></div>
    <p className="text-xs text-slate-400">C++ fits incoming and outgoing motion separately. Contact candidates remain unconfirmed; speed is a planar estimate. Select an event to inspect that exact recording frame.</p>
    {!graph.events.length&&<p className="text-sm text-slate-500">Waiting for a continuous accepted trajectory.</p>}
    <div className="max-h-72 overflow-auto space-y-2">{[...graph.events].reverse().map(event=><details key={event.id} className="rounded-lg border border-slate-800 p-3 text-xs font-mono">
      <summary className="cursor-pointer flex flex-wrap gap-3"><span className="text-slate-400">{event.timestamp.toFixed(3)} s</span><span className="text-cyan-200">{event.type.replaceAll('_',' ')}</span><span className="text-amber-300">{event.status}</span>{event.type==='STRIKE'&&<span>{event.outgoing_speed_kmh.toFixed(1)} km/h</span>}</summary>
      <p className="mt-3 text-slate-400">{event.reason}</p>
      {event.type==='STRIKE'&&<p className="my-2">In {event.incoming_speed_kmh.toFixed(1)} → out {event.outgoing_speed_kmh.toFixed(1)} km/h · heading change {event.heading_change_deg.toFixed(1)}° · fit residual {(event.fit_residual_m*100).toFixed(1)} cm · {event.player_side===null?'player unknown':'table end '+(event.player_side===0?'A':'B')}</p>}
      <button disabled={!canSeek} onClick={()=>sendCommand('SEEK_FRAME:'+event.frame_id)} className="mt-2 text-cyan-300 underline disabled:text-slate-600">Inspect frame {event.frame_id}</button>
    </details>)}</div>
    {!!graph.highlights.length&&<p className="text-xs text-amber-300">{graph.highlights.length} high-speed clip candidates identified. Event validation is required before treating these as verified highlights.</p>}
  </section>;
}
