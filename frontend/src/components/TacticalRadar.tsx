import React from 'react';
import type { TrajectorySample } from '../types/telemetry';

interface RadarProps {
  x: number; y: number; speedKmh?: number; isBounce?: boolean;
  impactPoints?: Array<{ x: number; y: number }>;
  hasBall?: boolean; frameId?: number;
  trajectory?: TrajectorySample[];
}

export default function TacticalRadar({ x, y, speedKmh = 0, impactPoints = [], hasBall = false, frameId = 0, trajectory=[] }: RadarProps) {
  const left = Math.min(0, hasBall ? x : 0) - .2;
  const top = Math.min(0, hasBall ? y : 0) - .2;
  const width = Math.max(1.525, hasBall ? x : 0) + .2 - left;
  const height = Math.max(2.74, hasBall ? y : 0) + .2 - top;
  return <div aria-label="2D table plane projection" data-frame-id={frameId} data-ball-x={hasBall ? x : ''} data-ball-y={hasBall ? y : ''} className="w-full p-4 bg-slate-950/80 rounded-2xl border border-slate-800 flex flex-col items-center">
    <div className="w-full text-xs font-mono text-cyan-300 mb-3">TOP-DOWN TABLE PROJECTION · FRAME {frameId}</div>
    <svg className="w-full h-[390px]" viewBox={`${left} ${top} ${width} ${height}`} role="img" aria-label="Ball projected onto the table plane">
      <rect x="0" y="0" width="1.525" height="2.74" fill="#115e59" stroke="#e2e8f0" strokeWidth=".012" />
      <path d="M .7625 0 V 2.74" stroke="#cbd5e1" strokeWidth=".006" />
      <path d="M -.08 1.37 H 1.605" stroke="#f8fafc" strokeWidth=".018" />
      <text x=".7625" y=".16" textAnchor="middle" fontSize=".07" fill="#a5f3fc">END A · y = 0 m</text>
      <text x=".7625" y="2.63" textAnchor="middle" fontSize=".07" fill="#a5f3fc">END B · y = 2.74 m</text>
      {impactPoints.filter(p => p.x >= 0 && p.x <= 1.525 && p.y >= 0 && p.y <= 2.74).map((p, i) => <circle key={i} cx={p.x} cy={p.y} r=".045" fill="#fb923c" opacity=".3" />)}
      {trajectory.map((s,i)=>{
        const p=s.table_position,previous=trajectory[i-1];if(!p||s.state==='LOST')return null;
        const color=s.state==='REACQUIRED'?'#34d399':s.accepted?'#22d3ee':'#fbbf24';
        return <g key={`${s.segment_id}:${s.frame_id}`}>
          {s.ellipse&&i%3===0&&<ellipse cx={p[0]} cy={p[1]} rx={s.ellipse[0]} ry={s.ellipse[1]} transform={`rotate(${s.ellipse[2]*180/Math.PI} ${p[0]} ${p[1]})`} fill={color} opacity=".08"/>}
          {previous?.table_position&&previous.segment_id===s.segment_id&&<line x1={previous.table_position[0]} y1={previous.table_position[1]} x2={p[0]} y2={p[1]} stroke={color} strokeWidth=".012" strokeDasharray={s.accepted?undefined:'.035 .025'}/>}
        </g>;
      })}
      {hasBall && <circle data-testid="radar-ball" cx={x} cy={y} r=".035" fill="#fbbf24" stroke="#fff7ed" strokeWidth=".012" />}
    </svg>
    <p className="text-xs text-slate-400 mt-3">{hasBall ? `x ${x.toFixed(2)} m · y ${y.toFixed(2)} m · ~${speedKmh.toFixed(1)} km/h` : 'Waiting for a detected ball and table surface'}</p>
    <p className="text-xs text-slate-500 mt-2">Same frame and projection as 3D. Airborne height is unknown; its image can project outside the table.</p>
  </div>;
}
