'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Play, RotateCcw, Zap, Sliders, Info } from 'lucide-react';
import { PhysicsParams, SpinType } from '../types/telemetry';

interface TrajectoryPoint {
  x: number; // m
  y: number; // m (height)
  z: number; // m (sideways)
  vx: number;
  vy: number;
  vz: number;
  time: number;
  isBounce?: boolean;
}

export const PhysicsSandbox: React.FC = () => {
  const [params, setParams] = useState<PhysicsParams>({
    speed_kmh: 85,
    launch_angle_deg: 12,
    side_angle_deg: 3,
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
  const [bouncePoint, setBouncePoint] = useState<{ x: number; z: number } | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // 3D Physics Integrator (Runge-Kutta 4th Order / Euler-Cromer)
  const computeTrajectory = (p: PhysicsParams): TrajectoryPoint[] => {
    const points: TrajectoryPoint[] = [];

    // Ball Constants
    const m = 0.0027; // 2.7 grams
    const r = 0.020;  // 20 mm radius
    const A = Math.PI * r * r;
    const rho = 1.225; // kg/m^3
    const g = 9.81;

    // Convert Initial Velocity
    const v0 = (p.speed_kmh * 1000) / 3600; // m/s
    const theta = (p.launch_angle_deg * Math.PI) / 180;
    const phi = (p.side_angle_deg * Math.PI) / 180;

    let vx = v0 * Math.cos(theta) * Math.cos(phi);
    let vy = v0 * Math.sin(theta); // Vertical (up)
    let vz = v0 * Math.cos(theta) * Math.sin(phi); // Sideways

    let x = 0;   // Distance along court (m)
    let y = 0.95; // Initial paddle height (m)
    let z = 0;   // Sideways offset (m)

    // Spin Vector w (rad/s)
    const omega = (p.spin_rpm * 2 * Math.PI) / 60;
    let wx = 0, wy = 0, wz = 0;

    if (p.spin_type === 'topspin') wz = -omega; // Pushes ball DOWN
    else if (p.spin_type === 'backspin') wz = omega; // Lifts ball UP
    else if (p.spin_type === 'sidespin_left') wy = omega; // Curves LEFT
    else if (p.spin_type === 'sidespin_right') wy = -omega; // Curves RIGHT

    const dt = 0.002; // 2 ms step size for ultra accuracy
    let t = 0;
    let bounced = false;
    let maxH = y;
    let bPt: { x: number; z: number } | null = null;

    for (let step = 0; step < 1500; step++) {
      points.push({ x, y, z, vx, vy, vz, time: t, isBounce: false });

      if (y > maxH) maxH = y;

      const vMag = Math.sqrt(vx * vx + vy * vy + vz * vz);
      if (vMag < 0.1 || x > 3.8 || y < 0) break;

      // 1. Drag Force: F_D = -0.5 * rho * A * C_D * |v| * v
      const fDragX = -0.5 * rho * A * p.drag_coefficient * vMag * vx;
      const fDragY = -0.5 * rho * A * p.drag_coefficient * vMag * vy;
      const fDragZ = -0.5 * rho * A * p.drag_coefficient * vMag * vz;

      // 2. Magnus Force: F_M = 0.5 * rho * A * C_M * r * (w x v)
      const crossX = wy * vz - wz * vy;
      const crossY = wz * vx - wx * vz;
      const crossZ = wx * vy - wy * vx;

      const fMagX = 0.5 * rho * A * p.magnus_coefficient * r * crossX;
      const fMagY = 0.5 * rho * A * p.magnus_coefficient * r * crossY;
      const fMagZ = 0.5 * rho * A * p.magnus_coefficient * r * crossZ;

      // Total Accelerations
      const ax = (fDragX + fMagX) / m;
      const ay = -g + (fDragY + fMagY) / m;
      const az = (fDragZ + fMagZ) / m;

      // Step Forward
      vx += ax * dt;
      vy += ay * dt;
      vz += az * dt;

      x += vx * dt;
      y += vy * dt;
      z += vz * dt;
      t += dt;

      // Table Bounce Check (Table surface at y = 0.76m, length x from 0.5m to 3.24m)
      const tableHeight = 0.76;
      if (!bounced && y <= tableHeight && x >= 0.5 && x <= 3.24 && Math.abs(z) <= 0.762) {
        bounced = true;
        y = tableHeight;
        vy = -vy * p.table_restitution; // Restitution coefficient
        vx *= (1 - p.table_friction);   // Surface friction speed loss
        vz *= (1 - p.table_friction);
        points[points.length - 1].isBounce = true;
        bPt = { x, z };
      }
    }

    setMaxHeight(Number(maxH.toFixed(2)));
    setFlightTime(Number(t.toFixed(2)));
    setBouncePoint(bPt);

    return points;
  };

  // Re-calculate when parameters change
  useEffect(() => {
    const pts = computeTrajectory(params);
    setTrajectory(pts);
  }, [params]);

  // Render 2D/3D Perspective Trajectory Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || trajectory.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width = canvas.clientWidth;
    const h = canvas.height = canvas.clientHeight;

    ctx.clearRect(0, 0, w, h);

    // Court Coordinates Mapping
    // Table dimensions: Length 2.74m (x: 0.5m to 3.24m), Height 0.76m, Net at x = 1.87m, height 0.1525m
    const scaleX = w / 4.0;
    const scaleY = h / 2.2;
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

    // Draw Table Base Block
    ctx.fillStyle = 'rgba(11, 83, 69, 0.4)';
    ctx.fillRect(tStart.cx, tStart.cy, tEnd.cx - tStart.cx, originY - tStart.cy);

    // Draw Table Surface Top Line
    ctx.beginPath();
    ctx.moveTo(tStart.cx, tStart.cy);
    ctx.lineTo(tEnd.cx, tEnd.cy);
    ctx.strokeStyle = '#00f3ff';
    ctx.lineWidth = 4;
    ctx.stroke();

    // Draw Net
    ctx.beginPath();
    ctx.moveTo(netPos.cx, netPos.cy);
    ctx.lineTo(netTop.cx, netTop.cy);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 3;
    ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Net Label
    ctx.fillStyle = '#ffffff';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillText('NET 15.25cm', netTop.cx - 28, netTop.cy - 8);

    // 2. Draw Trajectory Curve
    ctx.beginPath();
    trajectory.forEach((pt, idx) => {
      const { cx, cy } = toCanvas(pt.x, pt.y);
      if (idx === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    });

    const isTopspin = params.spin_type === 'topspin';
    ctx.strokeStyle = isTopspin ? '#00f3ff' : '#ff6600';
    ctx.lineWidth = 3.5;
    ctx.shadowColor = isTopspin ? '#00f3ff' : '#ff6600';
    ctx.shadowBlur = 12;
    ctx.stroke();

    // 3. Highlight Bounce Point
    if (bouncePoint) {
      const { cx, cy } = toCanvas(bouncePoint.x, 0.76);
      ctx.beginPath();
      ctx.arc(cx, cy, 8, 0, Math.PI * 2);
      ctx.fillStyle = '#ffe600';
      ctx.shadowColor = '#ffe600';
      ctx.shadowBlur = 18;
      ctx.fill();

      ctx.fillStyle = '#ffe600';
      ctx.font = 'bold 11px "JetBrains Mono", monospace';
      ctx.fillText(`BOUNCE (${bouncePoint.x.toFixed(2)}m)`, cx - 35, cy + 22);
    }
  }, [trajectory, params]);

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
            Simulates 3D aerodynamic forces (Magnus Effect F_M = 0.5 * rho * A * C_M * r/v * (w x v)) on standard 40mm ITTF table tennis balls.
          </p>
        </div>

        <div className="flex items-center gap-3 font-mono text-xs">
          <div className="bg-slate-900/90 border border-slate-800 px-3 py-1.5 rounded-lg text-slate-300">
            MAX HEIGHT: <span className="text-cyan-400 font-bold">{maxHeight} m</span>
          </div>
          <div className="bg-slate-900/90 border border-slate-800 px-3 py-1.5 rounded-lg text-slate-300">
            FLIGHT TIME: <span className="text-amber-400 font-bold">{flightTime} s</span>
          </div>
        </div>
      </div>

      {/* Main Grid: Visualizer + Control Sliders */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Columns: Trajectory Plotter Canvas */}
        <div className="lg:col-span-2 glass-panel p-4 rounded-2xl border border-slate-800 flex flex-col justify-between min-h-[420px]">
          <div className="flex items-center justify-between px-2 mb-2 font-mono text-xs text-slate-400">
            <span>SIDE ELEVATION PROFILE (X-Y PLANE)</span>
            <span className="text-cyan-400">MAGNUS DEFLECTION ACTIVE</span>
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
                  speed_kmh: 85,
                  launch_angle_deg: 12,
                  side_angle_deg: 3,
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
              min="20"
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
