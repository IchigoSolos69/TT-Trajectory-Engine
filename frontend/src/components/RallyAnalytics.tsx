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
  const maxSpeed = stats?.maxVelocity && stats.maxVelocity > 0 ? stats.maxVelocity : (currentFrame?.ball?.speed_kmh ?? currentFrame?.rally?.max_speed_kmh ?? 0);
  const elapsedTime = stats?.elapsedTimeSec && stats.elapsedTimeSec > 0 ? stats.elapsedTimeSec : (currentFrame?.rally?.rally_duration_s ?? 0);
  const spinRpm = currentFrame?.spin?.rpm ?? 0;

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 font-sans">
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
          <span className="text-xs font-mono text-slate-400 block uppercase">MAX VELOCITY</span>
          <span className="text-2xl font-black font-mono">
            <AnimatedNumber
              value={maxSpeed}
              decimals={1}
              suffix=" KM/H"
              className="text-amber-400 drop-shadow-[0_0_8px_rgba(251,191,36,0.5)]"
            />
          </span>
        </div>
        <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400">
          <Flame className="w-5 h-5" />
        </div>
      </div>

      {/* Metric 3: Active Spin RPM */}
      <div className="glass-panel p-4 rounded-xl border-slate-800 flex items-center justify-between hover:border-emerald-500/40 transition-colors">
        <div>
          <span className="text-xs font-mono text-slate-400 block uppercase">ILLUSTRATIVE SPIN (SIMULATED)</span>
          <span className="text-2xl font-black font-mono">
            <AnimatedNumber
              value={spinRpm}
              decimals={0}
              suffix=" RPM"
              className="text-emerald-400 drop-shadow-[0_0_8px_rgba(16,185,129,0.5)]"
            />
          </span>
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
  );
};
