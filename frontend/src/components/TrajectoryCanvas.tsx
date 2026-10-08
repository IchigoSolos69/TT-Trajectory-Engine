 'use client';

import React, { useRef, useLayoutEffect, useState } from 'react';
import { TelemetryFrame } from '../types/telemetry';
import { decodedFrames } from '../lib/decodedFrames';

interface TrajectoryCanvasProps {
  currentFrame: TelemetryFrame | null;
  historyLength?: number;
  showPlayerBoxes?: boolean;
  showTableCorners?: boolean;
  showVelocityVector?: boolean;
  showHeatmap?: boolean;
}

export const TrajectoryCanvas: React.FC<TrajectoryCanvasProps> = ({ currentFrame,
  historyLength = 25, showPlayerBoxes = true, showTableCorners = true,
  showVelocityVector = true, showHeatmap = false }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const pending = useRef<TelemetryFrame | null>(null);
  const options = useRef({ historyLength, showPlayerBoxes, showTableCorners, showVelocityVector, showHeatmap });
  options.current = { historyLength, showPlayerBoxes, showTableCorners, showVelocityVector, showHeatmap };
  const [displayed, setDisplayed] = useState<TelemetryFrame | null>(null);
  const [decodeError, setDecodeError] = useState(false);
  const memory = useRef<{last: TelemetryFrame | null; trail: Array<{x:number;y:number}>; impacts: Array<{x:number;y:number}>}>({last:null,trail:[],impacts:[]});
  useLayoutEffect(() => { pending.current = currentFrame; }, [currentFrame]);

  useLayoutEffect(() => {
    let stopped = false, busy = false;
    let {last,trail,impacts} = memory.current;
    const tick = () => {
      if (stopped) return;
      const f = pending.current;
      if (!f?.preview_jpeg) {
        if (last) {
          canvasRef.current?.getContext('2d')?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
          last = null; trail = []; impacts = []; setDisplayed(null);
          memory.current={last,trail,impacts};
        }
        return;
      }
      if (busy) return;
      busy = true;
      const cached=decodedFrames.get(f);
      const image = cached ?? new Image();
      const draw = () => {
        busy = false;
        if (stopped || !pending.current?.preview_jpeg) return;
        const canvas = canvasRef.current, ctx = canvas?.getContext('2d');
        if (!canvas || !ctx) return;
        // Commit image and all its annotations together. No independent video clock.
        canvas.width = image.width; canvas.height = image.height;
        canvas.dataset.frameId = String(f.frame_id);
        canvas.dataset.source = f.mode;
        const w = canvas.width, h = canvas.height;
        ctx.drawImage(image, 0, 0, w, h);
        const o = options.current;
        if (!last || last.mode !== f.mode || last.source_id !== f.source_id || f.frame_id < last.frame_id) {trail=[]; impacts=[];}
        if (!f.ball || (last && f.frame_id-last.frame_id > f.fps*.2)) trail=[];
        if (f.ball && f.frame_id !== last?.frame_id) {
          trail.push({x:f.ball.x,y:f.ball.y}); trail=trail.slice(-o.historyLength);
        }
        impacts=(f.image_impact_points??[]).map(([x,y])=>({x,y}));
        ctx.lineJoin='round'; ctx.lineCap='round';
        if(o.showTableCorners && f.table_corners?.length===4) {
          ctx.beginPath(); f.table_corners.forEach(([x,y],i)=>i?ctx.lineTo(x*w,y*h):ctx.moveTo(x*w,y*h)); ctx.closePath();
          ctx.strokeStyle='rgba(52,211,153,.9)';ctx.lineWidth=2;ctx.stroke();
        }
        if(o.showHeatmap) impacts.forEach(({x,y})=>{
          const gradient=ctx.createRadialGradient(x*w,y*h,0,x*w,y*h,22);
          gradient.addColorStop(0,'rgba(251,191,36,.5)');gradient.addColorStop(1,'rgba(251,191,36,0)');
          ctx.fillStyle=gradient;ctx.fillRect(x*w-22,y*h-22,44,44);
        });
        const label=(text:string,x:number,y:number,color:string)=>{
          ctx.font='bold 12px monospace'; const width=ctx.measureText(text).width+12;
          x=Math.max(0,Math.min(w-width,x));y=Math.max(20,Math.min(h,y));
          ctx.fillStyle='rgba(2,6,23,.85)';ctx.fillRect(x,y-18,width,21);
          ctx.fillStyle=color;ctx.fillText(text,x+6,y-3);
        };
        for(const detection of f.scene_detections ?? []) {
          if(detection.class_id===0 && !o.showPlayerBoxes) continue;
          if(detection.class_id===1 && !o.showTableCorners) continue;
          const [x,y,bw,bh]=detection.bbox;
          const color=detection.class_id===1?'#34d399':'#a78bfa';
          ctx.strokeStyle=color;ctx.lineWidth=1.5;
          ctx.strokeRect(x*w,y*h,bw*w,bh*h);
          label(`${detection.class_id===1?'TABLE':'PERSON / PLAYER'} ${(detection.confidence*100).toFixed(0)}%`,x*w,y*h,color);
        }
        ctx.strokeStyle='rgba(34,211,238,.8)';ctx.lineWidth=2;
        if(!f.intelligence){ctx.beginPath();trail.forEach(({x,y},i)=>i?ctx.lineTo(x*w,y*h):ctx.moveTo(x*w,y*h));ctx.stroke();}
        if(f.intelligence){
          const samples=f.intelligence.trajectory,sx=w/(f.source_width||w),sy=h/(f.source_height||h);
          for(let i=1;i<samples.length;i++){
            const s=samples[i],p=samples[i-1];if(s.state==='LOST'||p.state==='LOST'||s.segment_id!==p.segment_id)continue;
            ctx.setLineDash(s.accepted?[]:[5,5]);ctx.strokeStyle=s.state==='REACQUIRED'?'#34d399':s.accepted?'#22d3ee':'#fbbf24';
            ctx.beginPath();ctx.moveTo(p.pixel[0]*sx,p.pixel[1]*sy);ctx.lineTo(s.pixel[0]*sx,s.pixel[1]*sy);ctx.stroke();
          }
          ctx.setLineDash([]);
          const s=samples.at(-1);if(s&&s.state!=='LOST'){
            const x=s.pixel[0]*sx,y=s.pixel[1]*sy;ctx.strokeStyle=s.accepted?'#22d3ee':'#fbbf24';ctx.strokeRect(x-4,y-4,8,8);
            label(`${s.state} · ${s.reason.replaceAll('_',' ')}`,x+12,y+20,ctx.strokeStyle);
            if(s.state==='REJECTED'&&s.raw_observation){const rx=s.raw_observation[0]*sx,ry=s.raw_observation[1]*sy;ctx.strokeStyle='#f87171';ctx.beginPath();ctx.moveTo(rx-5,ry-5);ctx.lineTo(rx+5,ry+5);ctx.moveTo(rx+5,ry-5);ctx.lineTo(rx-5,ry+5);ctx.stroke();}
          }
        }
        if(f.ball) {
          const b=f.ball,x=b.x*w,y=b.y*h;
          ctx.strokeStyle='#fbbf24';ctx.lineWidth=2;ctx.beginPath();ctx.arc(x,y,8,0,Math.PI*2);ctx.stroke();
          label(`BALL ${(b.confidence*100).toFixed(0)}%${f.table_position?` · ~${b.speed_kmh.toFixed(1)} km/h`:''}`,x+12,y-10,'#fbbf24');
          if(o.showVelocityVector && trail.length>=2) {
            const prev=trail[trail.length-2], dx=x-prev.x*w,dy=y-prev.y*h;
            const scale=Math.min(3,65/Math.max(1,Math.hypot(dx,dy)));
            const ex=x+dx*scale,ey=y+dy*scale, angle=Math.atan2(dy,dx);
            ctx.strokeStyle='#fbbf24';ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(ex,ey);
            ctx.lineTo(ex-8*Math.cos(angle-.45),ey-8*Math.sin(angle-.45));ctx.moveTo(ex,ey);
            ctx.lineTo(ex-8*Math.cos(angle+.45),ey-8*Math.sin(angle+.45));ctx.stroke();
          }
        }
        last=f;setDisplayed(f);setDecodeError(false);
        memory.current={last,trail,impacts};
      };
      if(cached){draw();return;}
      image.onload=draw;
      image.onerror=()=>{busy=false;last=f;setDecodeError(true);};
      image.src=`data:image/jpeg;base64,${f.preview_jpeg}`;
    };
    tick();
    return()=>{stopped=true;};
  }, [currentFrame,historyLength,showPlayerBoxes,showTableCorners,showVelocityVector,showHeatmap]);

  return <div className="relative w-full h-full bg-black rounded-xl overflow-hidden border border-slate-800">
    <canvas ref={canvasRef} aria-label="Source footage with synchronized ball, person and table detections" className="w-full h-full object-contain" />
    {!displayed && <div className="absolute inset-0 flex items-center justify-center text-slate-400 text-sm font-mono">
      {currentFrame?.mode==='live'?'Start the selected camera to see live footage':'Waiting for source footage'}
    </div>}
    {displayed && <div className="absolute top-3 left-3 right-3 flex flex-wrap gap-2 text-[10px] font-mono pointer-events-none">
      <span className="bg-slate-950/85 text-cyan-300 px-3 py-2 rounded-lg">{displayed.mode==='live'?'LIVE CAMERA':displayed.source_video} · FRAME {displayed.frame_id}</span>
      <span className="bg-slate-950/85 text-amber-300 px-3 py-2 rounded-lg">{displayed.ball?'BALL DETECTED':'SEARCHING FOR BALL'}</span>
      <span className="bg-slate-950/85 text-emerald-300 px-3 py-2 rounded-lg">{displayed.table_corners?.length===4?'TABLE SURFACE ESTIMATED':'TABLE CALIBRATION UNAVAILABLE'}</span>
    </div>}
    {decodeError && <p role="alert" className="absolute bottom-3 left-3 text-rose-300 bg-slate-950 p-2">Source image could not be decoded.</p>}
    <div className="absolute bottom-3 left-3 bg-slate-950/80 px-3 py-2 rounded-lg text-[10px] text-slate-300 font-mono">Amber: ball · Violet: people · Green: table · Confidence is a model score</div>
  </div>;
};
