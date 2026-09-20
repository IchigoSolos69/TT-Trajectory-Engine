import React from 'react';
import { motion } from 'framer-motion';

interface RadarProps {
  x: number; // 0.0 to 1.525 meters
  y: number; // 0.0 to 2.74 meters
  speedKmh?: number;
  isBounce?: boolean;
  impactPoints?: Array<{ x: number; y: number }>;
}

export default function TacticalRadar({ x, y, speedKmh = 0, isBounce = false, impactPoints = [] }: RadarProps) {
  // Convert physical ITTF meters (1.525m width x 2.74m length) to CSS percentage coordinates
  const leftPercent = Math.max(0, Math.min(100, (x / 1.525) * 100));
  const topPercent = Math.max(0, Math.min(100, (y / 2.74) * 100));

  return (
    <div className="w-full max-w-lg mx-auto p-4 bg-slate-950/80 backdrop-blur-md rounded-2xl border border-slate-800 shadow-2xl flex flex-col items-center">
      {/* Header Info Banner */}
      <div className="w-full flex items-center justify-between mb-3 px-2 text-xs font-mono">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
          <span className="text-cyan-400 font-bold tracking-widest">2D TACTICAL RADAR & HEATMAP</span>
        </div>
        <div className="text-slate-400">
          X: <span className="text-cyan-300 font-bold">{x.toFixed(2)}m</span> | Y: <span className="text-cyan-300 font-bold">{y.toFixed(2)}m</span>
        </div>
      </div>

      {/* ITTF Standard Ratio Radar Canvas Container */}
      <div className="relative w-full aspect-[1.525/2.74] bg-slate-900/90 rounded-xl border-2 border-cyan-500/40 overflow-hidden shadow-[0_0_30px_rgba(6,182,212,0.15)]">
        {/* Table Surface Grid Lines */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:10%_10%] opacity-40" />

        {/* Outer Table Boundary Border */}
        <div className="absolute inset-2 border border-cyan-500/30 rounded-sm pointer-events-none" />

        {/* ITTF Net Line (Horizontal Midline) */}
        <div className="absolute top-1/2 left-0 w-full h-[2px] bg-white/60 -translate-y-1/2 shadow-[0_0_12px_rgba(255,255,255,0.8)] z-10" />
        <div className="absolute top-1/2 left-2 px-2 py-0.5 -translate-y-1/2 bg-slate-950/90 border border-slate-700 text-[9px] font-mono text-slate-300 rounded z-20">
          NET (y = 1.37m)
        </div>

        {/* ITTF Center Service Line (Vertical Midline) */}
        <div className="absolute top-0 left-1/2 w-[2px] h-full bg-white/20 -translate-x-1/2 z-0" />

        {/* Player Zone Indicators */}
        <div className="absolute top-3 left-1/2 -translate-x-1/2 text-[9px] font-mono tracking-widest text-slate-500 uppercase">
          PLAYER B (FAR ZONE)
        </div>
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2 text-[9px] font-mono tracking-widest text-slate-500 uppercase">
          PLAYER A (NEAR ZONE)
        </div>

        {/* Binned 2D Density Heatmap Overlay */}
        {(() => {
          // Bin impact points into 16 x 28 spatial grid cells (~10cm x 10cm over 1.525m x 2.74m table)
          const gridCols = 16;
          const gridRows = 28;
          const grid: number[][] = Array.from({ length: gridRows }, () => Array(gridCols).fill(0));
          let maxHits = 0;

          impactPoints.forEach((pt) => {
            if (pt.x < 0 || pt.x > 1.525 || pt.y < 0 || pt.y > 2.74) return; // Ignore out-of-bounds impacts
            const col = Math.max(0, Math.min(gridCols - 1, Math.floor((pt.x / 1.525) * gridCols)));
            const row = Math.max(0, Math.min(gridRows - 1, Math.floor((pt.y / 2.74) * gridRows)));
            grid[row][col] += 1;
            if (grid[row][col] > maxHits) maxHits = grid[row][col];
          });

          if (maxHits === 0) return null;

          const cells = [];
          for (let r = 0; r < gridRows; r++) {
            for (let c = 0; c < gridCols; c++) {
              const hits = grid[r][c];
              if (hits === 0) continue;
              const intensity = hits / maxHits; // 0.0 to 1.0
              const left = (c / gridCols) * 100;
              const top = (r / gridRows) * 100;
              const width = (1 / gridCols) * 100;
              const height = (1 / gridRows) * 100;

              let color = 'bg-cyan-400/50 shadow-[0_0_8px_rgba(6,182,212,0.6)]';
              if (intensity > 0.66) {
                color = 'bg-rose-500/80 shadow-[0_0_12px_rgba(244,63,94,0.9)]';
              } else if (intensity > 0.33) {
                color = 'bg-amber-400/70 shadow-[0_0_10px_rgba(251,191,36,0.8)]';
              }

              cells.push(
                <div
                  key={`${r}-${c}`}
                  className={`absolute rounded-sm transition-all ${color} pointer-events-none z-20`}
                  style={{
                    left: `${left}%`,
                    top: `${top}%`,
                    width: `${width}%`,
                    height: `${height}%`,
                  }}
                />
              );
            }
          }
          return cells;
        })()}

        {/* Impact Bounce Ripple Effect */}
        {isBounce && (
          <motion.div
            className="absolute w-8 h-8 rounded-full border-2 border-emerald-400"
            initial={{ scale: 0.2, opacity: 1 }}
            animate={{ scale: 2.5, opacity: 0 }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
            style={{
              left: `${leftPercent}%`,
              top: `${topPercent}%`,
              transform: 'translate(-50%, -50%)',
            }}
          />
        )}

        {/* Animated Trajectory Telemetry Ball */}
        <motion.div
          className="absolute w-4 h-4 bg-gradient-to-tr from-amber-500 to-orange-400 rounded-full shadow-[0_0_18px_rgba(249,115,22,0.9)] z-30"
          animate={{
            left: `${leftPercent}%`,
            top: `${topPercent}%`,
            x: '-50%',
            y: '-50%',
          }}
          transition={{
            type: 'spring',
            stiffness: 800,
            damping: 40,
            mass: 0.5,
          }}
        >
          {/* Inner Glow Core */}
          <div className="w-full h-full rounded-full bg-yellow-300 animate-pulse opacity-80" />
        </motion.div>
      </div>

      {/* Radar Footer Legend */}
      <div className="w-full mt-3 flex items-center justify-between text-[10px] font-mono text-slate-400 px-2">
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-amber-500" />
          Ball Position (2D Meters)
        </span>
        <span className="text-cyan-400 font-bold">
          Speed: {speedKmh > 0 ? speedKmh.toFixed(1) : '72.4'} km/h
        </span>
      </div>
    </div>
  );
}
