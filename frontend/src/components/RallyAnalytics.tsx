'use client';

import React from 'react';
import { Activity, Gauge, Flame, Timer } from 'lucide-react';
import { TelemetryFrame } from '../types/telemetry';
import { TelemetryStats } from '../hooks/useTelemetry';
import { AnimatedNumber } from './AnimatedNumber';

interface RallyAnalyticsProps {
  currentFrame: TelemetryFrame | null;
  stats?: TelemetryStats;
}

export const RallyAnalytics: React.FC<RallyAnalyticsProps> = ({ currentFrame, stats }) => {
  const strokeCount = stats?.rallyStrokes ?? (currentFrame?.rally?.stroke_count ?? 0);
  const maxSpeed = stats?.maxVelocity ?? currentFrame?.rally?.max_speed_kmh ?? 0;
  const elapsedTime = stats?.elapsedTimeSec && stats.elapsedTimeSec > 0 ? stats.elapsedTimeSec : (currentFrame?.rally?.rally_duration_s ?? 0);
  const spinRpm = currentFrame?.spin?.status==='estimated' ? currentFrame.spin.rpm : null;
  const strikes=currentFrame?.strikes ?? [];
  const lastStrike=strikes[strikes.length-1];

  return (
    <div className="space-y-3"><div className="grid grid-cols-1 md:grid-cols-4 gap-4 font-sans">
      {/* Metric 1: Stroke Count */}
      <div className="glass-panel p-4 rounded-xl border-slate-800 flex items-center justify-between hover:border-cyan-500/40 transition-colors">
        <div>
          <span className="text-xs font-mono text-slate-400 block uppercase">RALLY STROKES</span>
          <span className="text-2xl font-black text-white font-mono">
            <AnimatedNumber
              value={strokeCount}
              decimals={0}
              className="text-white drop-shadow-[0_0_8px_rgba(255,255,255,0.4)]"
            />
          </span>
        </div>
        <div className="p-3 rounded-xl bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
          <Activity className="w-5 h-5" />
        </div>
      </div>

      {/* Metric 2: Max Speed */}
      <div className="glass-panel p-4 rounded-xl border-slate-800 flex items-center justify-between hover:border-amber-500/40 transition-colors">
        <div>
          <span className="text-xs font-mono text-slate-400 block uppercase">LAST STRIKE · EST. SPEED</span>
          <span className="text-2xl font-black font-mono text-amber-400">{lastStrike?`${lastStrike.speed_kmh.toFixed(1)} KM/H`:'—'}</span>
          <span className="block text-[10px] text-slate-500 mt-1">Planar estimate · peak {maxSpeed.toFixed(1)} km/h</span>
        </div>
        <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400">
          <Flame className="w-5 h-5" />
        </div>
      </div>

      {/* Metric 3: Active Spin RPM */}
      <div className="glass-panel p-4 rounded-xl border-slate-800 flex items-center justify-between hover:border-emerald-500/40 transition-colors">
        <div>
          <span className="text-xs font-mono text-slate-400 block uppercase">OBSERVED ROTATION · EST.</span>
          <span className="text-2xl font-black font-mono">
            <span className="text-emerald-400">{spinRpm!==null?`~${spinRpm.toFixed(0)} RPM`:'UNRESOLVED'}</span>
          </span>
          <span className="block text-[10px] text-slate-500 mt-1 max-w-[220px]">{currentFrame?.spin?.reason ?? 'Waiting for visible ball markings'}</span>
        </div>
        <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
          <Gauge className="w-5 h-5" />
        </div>
      </div>

      {/* Metric 4: Rally Duration */}
      <div className="glass-panel p-4 rounded-xl border-slate-800 flex items-center justify-between hover:border-cyan-300/40 transition-colors">
        <div>
          <span className="text-xs font-mono text-slate-400 block uppercase">ELAPSED TIME</span>
          <span className="text-2xl font-black font-mono">
            <AnimatedNumber
              value={elapsedTime}
              decimals={1}
              suffix=" SEC"
              className="text-cyan-300 drop-shadow-[0_0_8px_rgba(103,232,249,0.5)]"
            />
          </span>
        </div>
        <div className="p-3 rounded-xl bg-cyan-500/10 border border-cyan-500/20 text-cyan-300">
          <Timer className="w-5 h-5" />
        </div>
      </div>
    </div>
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3 font-mono text-xs" aria-label="Strike speed history">
      <p className="text-slate-400 mb-2">STRIKE SPEEDS · estimated table-plane motion · detected events may include errors</p>
      {strikes.length ? <div className="flex gap-2 overflow-x-auto pb-1">{[...strikes].reverse().map(s=><div key={s.id} className="shrink-0 rounded-lg bg-slate-950 px-3 py-2 border border-amber-500/20">
        <span className="text-slate-400">#{s.id} · {s.source_time.toFixed(2)}s</span><strong className="block text-amber-300">{s.speed_kmh.toFixed(1)} km/h</strong><span className="text-[10px] text-slate-500">frame {s.frame_id}</span>
      </div>)}</div>:<p className="text-slate-500">Waiting for a detected strike and calibrated table.</p>}
    </div></div>
  );
};
