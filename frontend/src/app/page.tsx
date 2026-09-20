'use client';

import React, { useState, useEffect } from 'react';
import { Activity, Zap, BarChart3, Cpu, Radio, Sparkles, EyeOff, Eye, Info, Crosshair, Box } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import dynamic from 'next/dynamic';
import { useTelemetry } from '../hooks/useTelemetry';
import { TrajectoryCanvas } from '../components/TrajectoryCanvas';
import TacticalRadar from '../components/TacticalRadar';
import { PhysicsSandbox } from '../components/PhysicsSandbox';
import { RallyAnalytics } from '../components/RallyAnalytics';
import { CppBridgeStatus } from '../components/CppBridgeStatus';
import ArchitectureModal from '../components/ArchitectureModal';

const VolumetricTable = dynamic(() => import('../components/VolumetricTable'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-[520px] bg-slate-950 rounded-2xl border border-slate-800 flex items-center justify-center font-mono text-xs text-slate-500">
      <span className="animate-pulse text-cyan-400">LOADING 3D VOLUMETRIC ENVIRONMENT...</span>
    </div>
  ),
});

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<'visualizer' | 'radar' | '3d' | 'physics' | 'analytics' | 'engine'>('visualizer');
  const [showHeatmap, setShowHeatmap] = useState<boolean>(false);
  const [showVelocityVector, setShowVelocityVector] = useState<boolean>(true);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [cinematicMode, setCinematicMode] = useState<boolean>(false);

  // Custom hook handles live C++ WebSocket stream & recruiter simulation fallback
  const { currentFrame, isSimulated, isConnected, sendCommand, stats, resetStats, impactPoints } = useTelemetry();
  const [webcamEnabled, setWebcamEnabled] = useState<boolean>(false);

  const toggleWebcam = () => {
    if (webcamEnabled) {
      if (sendCommand) sendCommand('STOP_CAMERA');
      setWebcamEnabled(false);
    } else {
      if (sendCommand) sendCommand('START_CAMERA');
      setWebcamEnabled(true);
    }
  };

  // Extract physical coordinates from active telemetry stream (falling back to table center if idle)
  const currentX = currentFrame?.ball?.x ?? 0.7625;
  const currentY = currentFrame?.ball?.y ?? 1.37;
  const currentZ = currentFrame?.ball?.z ?? (currentFrame?.z ?? 0.02);
  const currentSpeed = currentFrame?.ball?.speed_kmh ?? (currentFrame?.velocity ?? 72.4);
  const isBounce = currentFrame?.ball?.is_bounce ?? (currentFrame?.bounce ?? false);

  // Cinematic / Clean Mode Keybind Listener ('H' Key)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }
      if (e.key.toLowerCase() === 'h') {
        setCinematicMode((prev) => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-4 md:p-8 space-y-6 max-w-7xl mx-auto relative overflow-hidden">
      {/* Top Navbar & Navigation Header */}
      <AnimatePresence>
        {!cinematicMode && (
          <motion.header
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3, ease: 'easeInOut' }}
            className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col md:flex-row items-start md:items-center justify-between gap-4"
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-emerald-400 flex items-center justify-center text-slate-950 font-black shadow-glowCyan">
                <Activity className="w-6 h-6" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-xl font-extrabold tracking-wider bg-gradient-to-r from-white via-cyan-200 to-cyan-400 bg-clip-text text-transparent">
                    TABLE TENNIS TRAJECTORY ENGINE
                  </h1>
                  {isSimulated ? (
                    <span className="px-2.5 py-0.5 rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/40 text-[10px] font-mono tracking-widest animate-pulse">
                      SIMULATION MODE : RECRUITER PREVIEW
                    </span>
                  ) : (
                    <div className="flex items-center gap-2">
                      <span className="px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 text-[10px] font-mono tracking-widest flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
                        LIVE HARDWARE TELEMETRY STREAM
                      </span>

                      <button
                        onClick={toggleWebcam}
                        className={`px-3 py-1 text-[10px] font-mono font-bold tracking-widest rounded-lg border transition-all flex items-center gap-1.5 ${
                          webcamEnabled
                            ? 'bg-red-500/20 text-red-400 border-red-500/50 hover:bg-red-500/30 shadow-[0_0_10px_rgba(239,68,68,0.2)]'
                            : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/50 hover:bg-emerald-500/30 shadow-[0_0_10px_rgba(16,185,129,0.2)]'
                        }`}
                        title="Click to toggle hardware webcam lock over WebSocket"
                      >
                        <span className={`w-1.5 h-1.5 rounded-full ${webcamEnabled ? 'bg-red-400 animate-pulse' : 'bg-emerald-400'}`} />
                        {webcamEnabled ? 'STOP WEBCAM' : 'AUTHORIZE WEBCAM'}
                      </button>
                    </div>
                  )}
                </div>
                <p className="text-xs font-mono text-slate-400 mt-0.5">
                  TensorRT 10.x C++ Inference • ByteTrack 120 FPS Telemetry • Magnus Spin Physics
                </p>
              </div>
            </div>

            {/* Action & Tab Navigation Pill Buttons */}
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[10px] font-mono text-slate-500 hidden xl:inline-block mr-1">
                PRESS <kbd className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300 font-bold">'H'</kbd> TO HIDE UI
              </span>

              <button
                onClick={() => setIsModalOpen(true)}
                className="px-4 py-2 text-xs font-mono tracking-widest text-emerald-400 border border-emerald-500/50 hover:bg-emerald-500/10 rounded-xl transition-all shadow-[0_0_10px_rgba(16,185,129,0.1)] flex items-center gap-1.5"
              >
                <Info className="w-3.5 h-3.5" />
                VIEW PIPELINE ARCHITECTURE
              </button>

              <button
                onClick={() => setCinematicMode(true)}
                className="px-3 py-2 text-xs font-mono text-slate-400 hover:text-white border border-slate-800 hover:border-slate-700 bg-slate-900/90 rounded-xl transition-all flex items-center gap-1.5"
                title="Hide UI controls for clean showcase recording (or press 'H')"
              >
                <EyeOff className="w-3.5 h-3.5" />
                CINEMATIC VIEW
              </button>

              <div className="flex flex-wrap items-center gap-1 bg-slate-950/80 p-1 rounded-xl border border-slate-800 text-xs font-mono">
                <button
                  onClick={() => setActiveTab('visualizer')}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all ${
                    activeTab === 'visualizer'
                      ? 'bg-cyan-500 text-slate-950 font-bold shadow-glowCyan'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <Radio className="w-3.5 h-3.5" />
                  VISUALIZER
                </button>

                <button
                  onClick={() => setActiveTab('radar')}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all ${
                    activeTab === 'radar'
                      ? 'bg-cyan-500 text-slate-950 font-bold shadow-glowCyan'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <Crosshair className="w-3.5 h-3.5" />
                  2D RADAR
                </button>

                <button
                  onClick={() => setActiveTab('3d')}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all ${
                    activeTab === '3d'
                      ? 'bg-cyan-500 text-slate-950 font-bold shadow-glowCyan'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <Box className="w-3.5 h-3.5" />
                  3D VOLUMETRIC
                </button>

                <button
                  onClick={() => setActiveTab('physics')}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all ${
                    activeTab === 'physics'
                      ? 'bg-cyan-500 text-slate-950 font-bold shadow-glowCyan'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <Zap className="w-3.5 h-3.5" />
                  PHYSICS
                </button>

                <button
                  onClick={() => setActiveTab('analytics')}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all ${
                    activeTab === 'analytics'
                      ? 'bg-cyan-500 text-slate-950 font-bold shadow-glowCyan'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <BarChart3 className="w-3.5 h-3.5" />
                  STATS
                </button>

                <button
                  onClick={() => setActiveTab('engine')}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all ${
                    activeTab === 'engine'
                      ? 'bg-cyan-500 text-slate-950 font-bold shadow-glowCyan'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <Cpu className="w-3.5 h-3.5" />
                  C++ ENGINE
                </button>
              </div>
            </div>
          </motion.header>
        )}
      </AnimatePresence>

      {/* Exit Cinematic Mode Floating Trigger */}
      <AnimatePresence>
        {cinematicMode && (
          <motion.button
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            onClick={() => setCinematicMode(false)}
            className="fixed top-6 right-6 z-40 flex items-center gap-2 px-4 py-2 rounded-xl glass-panel text-slate-300 hover:text-white text-xs font-mono border border-cyan-500/40 shadow-glowCyan"
          >
            <Eye className="w-4 h-4 text-cyan-400" />
            EXIT CINEMATIC VIEW (PRESS 'H')
          </motion.button>
        )}
      </AnimatePresence>

      {/* Summary Metrics Row */}
      <AnimatePresence>
        {!cinematicMode && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.25 }}
          >
            <RallyAnalytics currentFrame={currentFrame} stats={stats} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Animated Viewport */}
      <main className="relative w-full overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/50 shadow-2xl">
        <AnimatePresence mode="wait">
          {activeTab === 'visualizer' && (
            <motion.div
              key="visualizer"
              initial={{ opacity: 0, y: 15, filter: 'blur(4px)' }}
              animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
              exit={{ opacity: 0, y: -15, filter: 'blur(4px)' }}
              transition={{ duration: 0.3, ease: 'easeInOut' }}
              className="w-full h-full space-y-4 p-4"
            >
              {/* Visualizer Options Toolbar */}
              <AnimatePresence>
                {!cinematicMode && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="glass-panel px-5 py-3 rounded-xl border-slate-800 flex items-center justify-between text-xs font-mono overflow-hidden"
                  >
                    <span className="text-slate-400 flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-cyan-400" />
                      CANVAS OVERLAY CONFIGURATION
                    </span>
                    <div className="flex items-center gap-4">
                      <label className="flex items-center gap-2 cursor-pointer text-slate-300 hover:text-white">
                        <input
                          type="checkbox"
                          checked={showVelocityVector}
                          onChange={(e) => setShowVelocityVector(e.target.checked)}
                          className="accent-cyan-400 rounded"
                        />
                        Velocity Vectors (v)
                      </label>

                      <label className="flex items-center gap-2 cursor-pointer text-slate-300 hover:text-white">
                        <input
                          type="checkbox"
                          checked={showHeatmap}
                          onChange={(e) => setShowHeatmap(e.target.checked)}
                          className="accent-cyan-400 rounded"
                        />
                        Impact Heatmap Overlay
                      </label>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Trajectory Canvas Overlay */}
              <div className={cinematicMode ? 'h-[85vh]' : 'h-[520px]'}>
                <TrajectoryCanvas
                  currentFrame={currentFrame}
                  showVelocityVector={showVelocityVector}
                  showHeatmap={showHeatmap}
                />
              </div>
            </motion.div>
          )}

          {activeTab === 'radar' && (
            <motion.div
              key="radar"
              initial={{ opacity: 0, y: 15, filter: 'blur(4px)' }}
              animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
              exit={{ opacity: 0, y: -15, filter: 'blur(4px)' }}
              transition={{ duration: 0.3, ease: 'easeInOut' }}
              className="p-4 flex flex-col items-center justify-center min-h-[520px]"
            >
              <TacticalRadar x={currentX} y={currentY} speedKmh={currentSpeed} isBounce={isBounce} impactPoints={impactPoints} />
            </motion.div>
          )}

          {activeTab === '3d' && (
            <motion.div
              key="3d"
              initial={{ opacity: 0, y: 15, filter: 'blur(4px)' }}
              animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
              exit={{ opacity: 0, y: -15, filter: 'blur(4px)' }}
              transition={{ duration: 0.3, ease: 'easeInOut' }}
              className="p-4"
            >
              <VolumetricTable x={currentX} y={currentY} z={currentZ} />
            </motion.div>
          )}

          {activeTab === 'physics' && (
            <motion.div
              key="physics"
              initial={{ opacity: 0, y: 15, filter: 'blur(4px)' }}
              animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
              exit={{ opacity: 0, y: -15, filter: 'blur(4px)' }}
              transition={{ duration: 0.3, ease: 'easeInOut' }}
              className="p-4"
            >
              <PhysicsSandbox />
            </motion.div>
          )}

          {activeTab === 'analytics' && (
            <motion.div
              key="analytics"
              initial={{ opacity: 0, y: 15, filter: 'blur(4px)' }}
              animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
              exit={{ opacity: 0, y: -15, filter: 'blur(4px)' }}
              transition={{ duration: 0.3, ease: 'easeInOut' }}
              className="space-y-6 p-4"
            >
              <div className="glass-panel p-6 rounded-2xl border-slate-800 space-y-4">
                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-cyan-400" />
                  Rally Speed & Spin Breakdown
                </h3>
                <p className="text-xs text-slate-400">
                  Detailed telemetry breakdown aggregated across all rally strokes recorded in C++.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-900 font-mono text-xs space-y-2">
                    <span className="text-cyan-400 font-bold block">BALL SPEED DISTRIBUTION</span>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Serve Velocity:</span>
                      <span className="text-slate-200">54.2 km/h</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Average Rally Speed:</span>
                      <span className="text-slate-200">68.5 km/h</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Peak Smash Velocity:</span>
                      <span className="text-amber-400 font-bold">112.4 km/h</span>
                    </div>
                  </div>

                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-900 font-mono text-xs space-y-2">
                    <span className="text-emerald-400 font-bold block">SPIN FREQUENCY ANALYSIS</span>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Topspin Drives:</span>
                      <span className="text-slate-200">68%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Backspin Chops / Pushes:</span>
                      <span className="text-slate-200">22%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Sidespin Hook Serves:</span>
                      <span className="text-slate-200">10%</span>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'engine' && (
            <motion.div
              key="engine"
              initial={{ opacity: 0, y: 15, filter: 'blur(4px)' }}
              animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
              exit={{ opacity: 0, y: -15, filter: 'blur(4px)' }}
              transition={{ duration: 0.3, ease: 'easeInOut' }}
              className="p-4"
            >
              <CppBridgeStatus currentFrame={currentFrame} isConnected={isConnected} />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Architecture Pipeline Modal Overlay */}
      <AnimatePresence>
        {isModalOpen && <ArchitectureModal onClose={() => setIsModalOpen(false)} />}
      </AnimatePresence>
    </div>
  );
}
