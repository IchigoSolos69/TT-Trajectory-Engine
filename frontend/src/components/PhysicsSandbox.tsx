'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Play, RotateCcw, Zap, Sliders, Info } from 'lucide-react';
import { PhysicsParams, SpinType } from '../types/telemetry';
import { simulateFlight, FlightPoint as TrajectoryPoint } from '../lib/flightPhysics';

export const PhysicsSandbox: React.FC = () => {
  const [params, setParams] = useState<PhysicsParams>({
    speed_kmh: 18,
    launch_angle_deg: 15,
    side_angle_deg: 0,
    spin_rpm: 4800,
    spin_type: 'topspin',
    drag_coefficient: 0.40,
    table_restitution: 0.89,
    magnus_coefficient: 0.28,
    table_friction: 0.22,
  });

  const [trajectory, setTrajectory] = useState<TrajectoryPoint[]>([]);
  const [maxHeight, setMaxHeight] = useState<number>(0);
  const [flightTime, setFlightTime] = useState<number>(0);
  const [bouncePoint, setBouncePoint] = useState<TrajectoryPoint[]>([]);

  // Net Status State
  const [netStatus, setNetStatus] = useState<{
    checked: boolean;
    hit: boolean;
    heightAtNet: number;
    clearanceCm: number;
    hitPoint?: { x: number; y: number };
  }>({ checked: false, hit: false, heightAtNet: 0, clearanceCm: 0 });

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const result = simulateFlight(params);
    setTrajectory(result.points);
    setMaxHeight(result.maxHeight);
    setFlightTime(result.flightTime);
    setBouncePoint(result.bounces);
    setNetStatus(result.net);
  }, [params]);

  // Render 2D Elevation Profile Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || trajectory.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = (canvas.width = canvas.clientWidth);
    const h = (canvas.height = canvas.clientHeight);

    ctx.clearRect(0, 0, w, h);

    // Court Coordinates Mapping
    // Table dimensions: Length 2.74m (x: 0.5m to 3.24m), Height 0.76m, Net at x = 1.87m, height 0.1525m
    const scaleX = (w - 80) / Math.max(4, ...trajectory.map(p => p.x));
    const scaleY = (h - 80) / Math.max(1.4, maxHeight + .15);
    const originX = 40;
    const originY = h - 50;

    const toCanvas = (x: number, y: number) => ({
      cx: originX + x * scaleX,
      cy: originY - y * scaleY,
    });

    // 1. Floor & Table Surface
    const tStart = toCanvas(0.5, 0.76);
    const tEnd = toCanvas(3.24, 0.76);
    const netPos = toCanvas(1.87, 0.76);
    const netTop = toCanvas(1.87, 0.76 + 0.1525);

    ctx.fillStyle = '#334155';
    ctx.fillRect(tStart.cx, tStart.cy, tEnd.cx-tStart.cx, 7);
    for (const leg of [.8, 2.94]) {
      const pt = toCanvas(leg, .76);
      ctx.fillRect(pt.cx-3, pt.cy+7, 6, originY-pt.cy-7);
    }
    ctx.strokeStyle = '#334155'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(20, originY); ctx.lineTo(w-20, originY); ctx.stroke();
    ctx.fillStyle = '#94a3b8'; ctx.font = '11px monospace';
    ctx.fillText('Height above floor (m) · side elevation · ball radius 20 mm', 20, 20);
    ctx.fillText('Table length 2.74 m', tStart.cx, originY+22);

    // Draw Table Surface Top Line
    ctx.beginPath();
    ctx.moveTo(tStart.cx, tStart.cy);
    ctx.lineTo(tEnd.cx, tEnd.cy);
    ctx.strokeStyle = '#00f3ff';
    ctx.lineWidth = 4;
    ctx.stroke();

    // Draw Net (White dashed line + solid top band)
    ctx.beginPath();
    ctx.moveTo(netPos.cx, netPos.cy);
    ctx.lineTo(netTop.cx, netTop.cy);
    ctx.strokeStyle = netStatus.hit ? '#f43f5e' : '#ffffff';
    ctx.lineWidth = netStatus.hit ? 4 : 3;
    ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Net Top Band Line
    ctx.beginPath();
    ctx.arc(netTop.cx, netTop.cy, 3, 0, Math.PI * 2);
    ctx.fillStyle = netStatus.hit ? '#f43f5e' : '#ffffff';
    ctx.fill();

    // Net Label
    ctx.fillStyle = netStatus.hit ? '#f43f5e' : '#ffffff';
    ctx.font = 'bold 10px "JetBrains Mono", monospace';
    ctx.fillText('NET 15.25cm', netTop.cx - 28, netTop.cy - 8);

    // 2. Draw Trajectory Curve
    ctx.beginPath();
    trajectory.forEach((pt, idx) => {
      const { cx, cy } = toCanvas(pt.x, pt.y);
      if (idx === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    });

    const isTopspin = params.spin_type === 'topspin';
    ctx.strokeStyle = netStatus.hit ? '#f43f5e' : (isTopspin ? '#00f3ff' : '#ff6600');
    ctx.lineWidth = 3.5;
    ctx.shadowColor = netStatus.hit ? '#f43f5e' : (isTopspin ? '#00f3ff' : '#ff6600');
    ctx.shadowBlur = 12;
    ctx.stroke();

    // 3. Highlight Net Collision Point if Hit
    if (netStatus.hit && netStatus.hitPoint) {
      const { cx, cy } = toCanvas(netStatus.hitPoint.x, netStatus.hitPoint.y);
      ctx.beginPath();
      ctx.arc(cx, cy, 9, 0, Math.PI * 2);
      ctx.fillStyle = '#f43f5e';
      ctx.shadowColor = '#f43f5e';
      ctx.shadowBlur = 20;
      ctx.fill();

      ctx.fillStyle = '#f43f5e';
      ctx.font = 'bold 11px "JetBrains Mono", monospace';
      ctx.fillText(`NET HIT (${netStatus.heightAtNet.toFixed(2)}m)`, cx - 40, cy - 14);
    }

    bouncePoint.forEach((bounce, index) => {
      const { cx, cy } = toCanvas(bounce.x, bounce.y);
      ctx.shadowBlur = 0;
      ctx.fillStyle = index === 0 ? '#fbbf24' : '#a78bfa';
      ctx.beginPath(); ctx.arc(cx, cy, 6, 0, Math.PI*2); ctx.fill();
      ctx.font = 'bold 11px monospace';
      ctx.fillText(`BOUNCE ${index+1} · ${(bounce.x-.5).toFixed(2)} m`, Math.min(w-155, Math.max(10,cx-40)), cy+25+(index%2)*17);
    });

  }, [trajectory, params, netStatus, bouncePoint, maxHeight]);

  return (
    <div className="w-full space-y-6">
      {/* Sandbox Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 glass-panel p-6 rounded-2xl border-cyan-500/20">
        <div>
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-cyan-400" />
            <h2 className="text-xl font-bold text-white tracking-wide">
              Ping Pong Aerodynamics & Magnus Effect Simulator
            </h2>
          </div>
          <p className="text-slate-400 text-xs mt-1">
            Simulates 3D aerodynamic forces (Magnus Effect F_M = 0.5 * rho * A * C_M * r * (w x v)) with 15.25cm Net Collision Detection.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 font-mono text-xs">
          {netStatus.hit ? (
            <div className="bg-rose-500/20 border border-rose-500/50 px-3 py-1.5 rounded-lg text-rose-300 flex items-center gap-1.5 shadow-[0_0_12px_rgba(244,63,94,0.3)]">
              <span className="w-2 h-2 rounded-full bg-rose-400 animate-ping" />
              NET COLLISION (ball centre {netStatus.heightAtNet.toFixed(2)} m)
            </div>
          ) : (
            <div className="bg-emerald-500/20 border border-emerald-500/50 px-3 py-1.5 rounded-lg text-emerald-300 flex items-center gap-1.5 shadow-[0_0_12px_rgba(16,185,129,0.3)]">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              {netStatus.checked ? `CLEARED NET (${netStatus.clearanceCm.toFixed(1)}cm ball clearance)` : 'NET NOT CROSSED'}
            </div>
          )}

          <div className="bg-slate-900/90 border border-slate-800 px-3 py-1.5 rounded-lg text-slate-300">
            MAX HEIGHT: <span className="text-cyan-400 font-bold">{maxHeight.toFixed(2)} m</span>
          </div>
          <div className="bg-slate-900/90 border border-slate-800 px-3 py-1.5 rounded-lg text-slate-300">
            FLIGHT TIME: <span className="text-amber-400 font-bold">{flightTime.toFixed(2)} s</span>
          </div>
        </div>
      </div>

      {/* Main Grid: Visualizer + Control Sliders */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Columns: Trajectory Plotter Canvas */}
        <div className="lg:col-span-2 glass-panel p-4 rounded-2xl border border-slate-800 flex flex-col justify-between min-h-[420px]">
          <div className="flex items-center justify-between px-2 mb-2 font-mono text-xs text-slate-400">
            <span>SIDE ELEVATION PROFILE (X-Y PLANE)</span>
            <span className="text-cyan-400">{bouncePoint.length} TABLE BOUNCES · ESTIMATED PHYSICS</span>
          </div>

          <canvas
            ref={canvasRef}
            className="w-full h-[360px] block rounded-xl bg-slate-950 border border-slate-900 cursor-crosshair"
          />
        </div>

        {/* Right 1 Column: Physics Parameters Panel */}
        <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-5 font-sans">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="font-semibold text-white text-sm flex items-center gap-2">
              <Sliders className="w-4 h-4 text-cyan-400" />
              Ball Mechanics Controls
            </h3>
            <button
              onClick={() =>
                setParams({
                  speed_kmh: 18,
                  launch_angle_deg: 15,
                  side_angle_deg: 0,
                  spin_rpm: 4800,
                  spin_type: 'topspin',
                  drag_coefficient: 0.40,
                  table_restitution: 0.89,
                  magnus_coefficient: 0.28,
                  table_friction: 0.22,
                })
              }
              className="text-slate-400 hover:text-white transition-colors"
              title="Reset Defaults"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

          {/* Spin Type Radio Selector */}
          <div>
            <label className="text-xs font-mono text-slate-400 block mb-2">SPIN TYPE</label>
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              {(['topspin', 'backspin', 'sidespin_left', 'sidespin_right'] as SpinType[]).map((st) => (
                <button
                  key={st}
                  onClick={() => setParams({ ...params, spin_type: st })}
                  className={`py-2 px-3 rounded-lg border text-center transition-all ${
                    params.spin_type === st
                      ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300 font-bold shadow-glowCyan'
                      : 'bg-slate-900 border-slate-800 text-slate-400 hover:border-slate-700'
                  }`}
                >
                  {st.replace('_', ' ').toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          {/* Speed Slider */}
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs font-mono">
              <span className="text-slate-400">INITIAL SPEED (v₀)</span>
              <span className="text-cyan-400 font-bold">{params.speed_kmh} KM/H</span>
            </div>
            <input
              type="range"
              min="8"
              max="135"
              value={params.speed_kmh}
              onChange={(e) => setParams({ ...params, speed_kmh: Number(e.target.value) })}
              className="w-full accent-cyan-400 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>

          {/* Spin RPM Slider */}
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs font-mono">
              <span className="text-slate-400">SPIN RATE (ω)</span>
              <span className="text-amber-400 font-bold">{params.spin_rpm} RPM</span>
            </div>
            <input
              type="range"
              min="0"
              max="9000"
              step="100"
              value={params.spin_rpm}
              onChange={(e) => setParams({ ...params, spin_rpm: Number(e.target.value) })}
              className="w-full accent-amber-400 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>

          {/* Launch Angle Slider */}
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs font-mono">
              <span className="text-slate-400">LAUNCH ANGLE (θ)</span>
              <span className="text-emerald-400 font-bold">{params.launch_angle_deg}°</span>
            </div>
            <input
              type="range"
              min="-10"
              max="45"
              value={params.launch_angle_deg}
              onChange={(e) => setParams({ ...params, launch_angle_deg: Number(e.target.value) })}
              className="w-full accent-emerald-400 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>

          {/* Table Restitution (Bounce Coefficient) */}
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs font-mono">
              <span className="text-slate-400">BOUNCE RESTITUTION (e)</span>
              <span className="text-purple-400 font-bold">{params.table_restitution}</span>
            </div>
            <input
              type="range"
              min="0.50"
              max="0.98"
              step="0.01"
              value={params.table_restitution}
              onChange={(e) => setParams({ ...params, table_restitution: Number(e.target.value) })}
              className="w-full accent-purple-400 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>
        </div>
      </div>
    </div>
  );
};
