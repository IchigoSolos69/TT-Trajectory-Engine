'use client';

import React from 'react';
import { Cpu, Server, Wifi, CheckCircle2, ShieldCheck, Zap } from 'lucide-react';
import { TelemetryFrame } from '../types/telemetry';

interface CppBridgeStatusProps {
  currentFrame: TelemetryFrame | null;
  isConnected: boolean;
}

export const CppBridgeStatus: React.FC<CppBridgeStatusProps> = ({ currentFrame, isConnected }) => {
  return (
    <div className="glass-panel p-6 rounded-2xl border-slate-800 space-y-6 font-sans">
      {/* Title Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white tracking-wide">
              C++ TensorRT Engine & Telemetry Bridge
            </h3>
            <p className="text-xs text-slate-400 font-mono">
              OpenCV Capture → TensorRT 10.x CUDA Stream → ByteTrack Tracker → uWebSockets (Port 9002)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs px-3 py-1.5 rounded-full border bg-slate-900 border-slate-700">
          <span className={`w-2.5 h-2.5 rounded-full ${isConnected ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
          <span className={isConnected ? 'text-emerald-400 font-bold' : 'text-amber-400'}>
            {isConnected ? 'WEBSOCKET ACTIVE (uWebSockets)' : 'SIMULATION MODE (FALLBACK)'}
          </span>
        </div>
      </div>

      {/* Experimental Bounce Detector Alert Banner */}
      <div className="flex items-center justify-between p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 font-mono text-xs shadow-[0_0_15px_rgba(245,158,11,0.1)]">
        <div className="flex items-center gap-2.5">
          <span className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-ping" />
          <span className="font-bold text-amber-400 uppercase tracking-wider">BOUNCE DETECTOR STATUS:</span>
          <span className="text-slate-200">
            {currentFrame?.bounce_detector_status || 'EXPERIMENTAL (~15.5% F1 - Mode B Gated Raw Top-1 Feed)'}
          </span>
        </div>
        <span className="text-[10px] text-amber-400/80 font-semibold px-2 py-0.5 rounded bg-amber-500/20 border border-amber-500/30">
          ACTIVE-RALLY GATE ROADMAP
        </span>
      </div>

      {/* Grid of Hardware & Pipeline Specifications */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono">
        {/* Card 1: CUDA & TensorRT Buffer Allocation */}
        <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between text-slate-300">
            <span className="flex items-center gap-1.5 text-cyan-400 font-bold">
              <Zap className="w-3.5 h-3.5" /> CUDA MEMORY ALLOCATION
            </span>
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
          </div>
          <div className="text-slate-400 space-y-1 pt-1">
            <div className="flex justify-between">
              <span>INPUT BUFFER (CHW):</span>
              <span className="text-slate-200">1 x 3 x 640 x 640 (FP32)</span>
            </div>
            <div className="flex justify-between">
              <span>OUTPUT BUFFER (NMS):</span>
              <span className="text-slate-200">1 x 84 x 8400 (Float)</span>
            </div>
            <div className="flex justify-between">
              <span>EXECUTION CONTEXT:</span>
              <span className="text-emerald-400">cudaStreamAsync</span>
            </div>
          </div>
        </div>

        {/* Card 2: Inference Throughput & Latency */}
        <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between text-slate-300">
            <span className="flex items-center gap-1.5 text-amber-400 font-bold">
              <Server className="w-3.5 h-3.5" /> INFERENCE PERFORMANCE
            </span>
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
          </div>
          <div className="text-slate-400 space-y-1 pt-1">
            <div className="flex justify-between">
              <span>TARGET FPS:</span>
              <span className="text-cyan-400 font-bold">120 FPS</span>
            </div>
            <div className="flex justify-between">
              <span>CURRENT FPS:</span>
              <span className="text-slate-200">{currentFrame?.fps || 120} FPS</span>
            </div>
            <div className="flex justify-between">
              <span>LATENCY (GPU INFER):</span>
              <span className="text-emerald-400">~2.4 ms / frame</span>
            </div>
          </div>
        </div>

        {/* Card 3: Object Tracking Strategy */}
        <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between text-slate-300">
            <span className="flex items-center gap-1.5 text-emerald-400 font-bold">
              <ShieldCheck className="w-3.5 h-3.5" /> OBJECT TRACKER
            </span>
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
          </div>
          <div className="text-slate-400 space-y-1 pt-1">
            <div className="flex justify-between">
              <span>TRACKER ALGORITHM:</span>
              <span className="text-slate-200 font-bold">ByteTrack C++</span>
            </div>
            <div className="flex justify-between">
              <span>ASSOCIATION:</span>
              <span className="text-slate-200">IoU + Kalman Filter</span>
            </div>
            <div className="flex justify-between">
              <span>HIGH-SPEED ADVANTAGE:</span>
              <span className="text-cyan-400">Zero Feature Overhead</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
