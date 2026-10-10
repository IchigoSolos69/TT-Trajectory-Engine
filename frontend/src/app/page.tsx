'use client';

import React, { useState, useEffect } from 'react';
import { Activity, Zap, BarChart3, Cpu, Radio, Sparkles, EyeOff, Eye, Info, Crosshair, Box } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import dynamic from 'next/dynamic';
import { useTelemetry } from '../hooks/useTelemetry';
import { TrajectoryCanvas } from '../components/TrajectoryCanvas';
import { TrainingCaptureControls } from '../components/TrainingCaptureControls';
import TacticalRadar from '../components/TacticalRadar';
import { PhysicsSandbox } from '../components/PhysicsSandbox';
import { RallyAnalytics } from '../components/RallyAnalytics';
import { CppBridgeStatus } from '../components/CppBridgeStatus';
import ArchitectureModal from '../components/ArchitectureModal';
import { spatialFrame } from '../lib/spatialFrame';
import { MeasurementQuality } from '../components/MeasurementQuality';
import { RallyTimeline } from '../components/RallyTimeline';

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
  const [showForecast, setShowForecast] = useState<boolean>(false);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [cinematicMode, setCinematicMode] = useState<boolean>(false);

  // Both source modes receive measured telemetry from the C++ backend.
  const { currentFrame, isSimulated, isConnected, sendCommand, stats, resetStats, impactPoints } = useTelemetry();
  const [webcamEnabled, setWebcamEnabled] = useState<boolean>(false);
  const [selectedCamera, setSelectedCamera] = useState<number>(0);
  const [recordingPickerError,setRecordingPickerError]=useState<string|null>(null);
  const openRecording=async()=>{
    setRecordingPickerError(null);
    try{if(window.ttDesktop)await window.ttDesktop.openRecording();else sendCommand('OPEN_VIDEO_DIALOG');}
    catch(error){setRecordingPickerError(error instanceof Error?error.message:String(error));}
  };
  useEffect(() => {
    setWebcamEnabled(!!currentFrame?.camera_active);
  }, [currentFrame]);

  const toggleWebcam = () => {
    if (!isConnected || currentFrame?.mode !== 'live') return;
    if (webcamEnabled) {
      if (sendCommand) sendCommand('STOP_CAMERA');
      setWebcamEnabled(false);
    } else {
      if (sendCommand) sendCommand('START_CAMERA');
      setWebcamEnabled(true);
    }
  };

  const handleSelectCamera = (idx: number) => {
    if (!isConnected || currentFrame?.mode !== 'live') return;
    setSelectedCamera(idx);
    if (sendCommand) {
      sendCommand(`SET_CAMERA:${idx}`);
    }
  };

  // Extract physical coordinates from active telemetry stream (falling back to table center if idle)
  const spatial = spatialFrame(currentFrame);
  const currentX = spatial.x;
  const currentY = spatial.y;
  const currentSpeed = currentFrame?.ball?.speed_kmh ?? (currentFrame?.velocity ?? 0);
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
                  {isConnected ? (
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 text-[10px] font-mono tracking-widest flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
                        {isSimulated ? 'DATASET VIDEO ANALYSIS' : 'LIVE HARDWARE TELEMETRY STREAM'}
                      </span>

                      {/* Dynamic Camera Selector Buttons */}
                      {currentFrame?.mode === 'live' && <>
                      <div className="flex items-center gap-1 bg-slate-900/90 border border-slate-800 rounded-lg p-0.5 font-mono text-[9px]">
                        <button
                          onClick={() => handleSelectCamera(0)}
                          disabled={currentFrame?.mode !== 'live'}
                          className={`px-2 py-0.5 rounded font-bold transition-all ${
                            selectedCamera === 0
                              ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-[0_0_8px_rgba(6,182,212,0.2)]'
                              : 'text-slate-400 hover:text-slate-200'
                          }`}
                          title="Select Camera Index 0 (Built-in Laptop Webcam)"
                        >
                          CAM 0 (BUILT-IN)
                        </button>
                        <button
                          onClick={() => handleSelectCamera(1)}
                          disabled={currentFrame?.mode !== 'live'}
                          className={`px-2 py-0.5 rounded font-bold transition-all ${
                            selectedCamera === 1
                              ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-[0_0_8px_rgba(6,182,212,0.2)]'
                              : 'text-slate-400 hover:text-slate-200'
                          }`}
                          title="Select Camera Index 1 (External USB Table Camera)"
                        >
                          CAM 1 (USB EXT)
                        </button>
                      </div>

                      <button
                        onClick={toggleWebcam}
                        disabled={currentFrame?.mode !== 'live'}
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
                      </>}
                    </div>
                  ) : (
                    <div className="flex items-center gap-2">
                      <span className="px-2.5 py-0.5 rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/40 text-[10px] font-mono tracking-widest flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
                        DISCONNECTED : WAITING FOR ENGINE
                      </span>
                      <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-[9px] font-mono text-slate-400">
                        CONNECTING TO LOCAL ENGINE…
                      </span>
                    </div>
                  )}
                </div>
                <p className="text-xs font-mono text-slate-400 mt-0.5">
                  TensorRT C++ Inference • ByteTrack source-frame tracking • Magnus Spin Physics
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
      {!cinematicMode && <MeasurementQuality data={currentFrame?.intelligence}/>}
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
                      FOOTAGE & DETECTIONS
                    </span>
                    <div className="flex items-center gap-4 flex-wrap">

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
                      {currentFrame?.ball_model?.backend==='blurball'&&<label className="flex items-center gap-2 cursor-pointer text-violet-200"><input aria-label="Experimental trajectory forecast" type="checkbox" checked={showForecast} onChange={e=>setShowForecast(e.target.checked)} className="accent-violet-400"/>Trajectory forecast · experimental</label>}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {isSimulated && <div className="flex flex-wrap items-center gap-3 font-mono text-xs px-2" aria-label="Dataset playback controls">
                <button disabled={currentFrame?.video_picker_open} className="px-3 py-2 rounded-lg border border-cyan-600 text-cyan-300 disabled:opacity-50" onClick={openRecording}>{currentFrame?.video_picker_open ? 'File picker open…' : 'Open recording…'}</button>
                <button className="px-4 py-2 rounded-lg bg-cyan-500 text-slate-950 font-bold" onClick={()=>sendCommand(currentFrame?.paused?'PLAY':'PAUSE')}>{currentFrame?.paused?'Play':'Pause'}</button>
                <button className="px-3 py-2 rounded-lg border border-slate-700" onClick={()=>sendCommand('RESTART')}>Restart</button>
                {currentFrame?.analysis && <><button aria-label="Previous frame" className="px-3 py-2 rounded-lg border border-slate-700" onClick={()=>sendCommand('STEP_BACK')}>−1 frame</button><button aria-label="Next frame" className="px-3 py-2 rounded-lg border border-slate-700" onClick={()=>sendCommand('STEP_FORWARD')}>+1 frame</button></>}
                <label className="flex items-center gap-2">Playback speed
                  <select aria-label="Playback speed" className="bg-slate-900 border border-slate-700 rounded-lg p-2" value={currentFrame?.playback_rate ?? 1} onChange={e=>sendCommand(`PLAYBACK_RATE:${e.target.value}`)}>
                    {[.25,.5,1,1.5,2,4].map(rate=><option key={rate} value={rate}>{rate}×</option>)}
                  </select>
                </label>
                <span className="text-slate-400">{(currentFrame?.source_time??0).toFixed(2)} s / {((currentFrame?.total_frames??0)/(currentFrame?.fps||1)).toFixed(1)} s {currentFrame?.ended?'· End of footage':''}</span>
                {currentFrame?.analysis ? <div className="w-full space-y-2" aria-label="Recording analysis progress">
                  <p className="text-emerald-300">{currentFrame.analysis.status} · {currentFrame.analysis.analyzed_frames.toLocaleString()} / {currentFrame.analysis.total_frames.toLocaleString()} frames analyzed · no analytical frame skipping</p>
                  <progress className="w-full h-1 accent-emerald-400" value={currentFrame.analysis.analyzed_frames} max={currentFrame.analysis.total_frames}/>
                  <label className="flex gap-3 items-center text-slate-400">Seek frame <input aria-label="Seek recording frame" className="flex-1 accent-cyan-400" type="range" min={1} max={currentFrame.analysis.total_frames} value={currentFrame.analysis.requested_frame||currentFrame.frame_id||1} onChange={e=>sendCommand(`SEEK_FRAME:${e.target.value}`)}/><span>{currentFrame.frame_id}</span></label>
                  {currentFrame.analysis.waiting_for_analysis&&<p role="status" className="text-amber-300">Waiting for analysis{currentFrame.analysis.requested_frame?` of frame ${currentFrame.analysis.requested_frame}`:''}… Playback does not skip analysis.</p>}
                </div> : <span className="text-slate-500">{(currentFrame?.processing_fps??0).toFixed(0)} analyzed FPS · {currentFrame?.skipped_frames??0} frames skipped to keep pace</span>}
              </div>}
              {currentFrame?.video_error && <p role="alert" className="text-sm text-amber-300 px-2">{currentFrame.video_error}</p>}
              {recordingPickerError && <p role="alert" className="text-sm text-amber-300 px-2">{recordingPickerError}</p>}

              {currentFrame?.mode === 'live' && <TrainingCaptureControls frame={currentFrame} sendCommand={sendCommand} />}

              {/* Trajectory Canvas Overlay */}
              <div className={cinematicMode ? 'h-[85vh]' : 'h-[520px]'}>
                <TrajectoryCanvas
                  currentFrame={currentFrame}
                  showVelocityVector={showVelocityVector}
                  showHeatmap={showHeatmap}
                  showForecast={showForecast}
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
              className="p-4 grid xl:grid-cols-2 gap-4 min-h-[520px]"
            >
              <div className="h-[520px]"><TrajectoryCanvas currentFrame={currentFrame} showVelocityVector={showVelocityVector} showHeatmap={showHeatmap} showForecast={showForecast}/></div>
              <TacticalRadar x={currentX} y={currentY} speedKmh={currentSpeed} isBounce={isBounce} impactPoints={impactPoints} hasBall={spatial.available} frameId={spatial.frameId} analysisId={currentFrame?.analysis_id} trajectory={currentFrame?.intelligence?.trajectory} />
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
              <div className="h-[300px] mb-4"><TrajectoryCanvas currentFrame={currentFrame} showVelocityVector={showVelocityVector} showHeatmap={showHeatmap} showForecast={showForecast}/></div>
              <VolumetricTable x={currentX} y={currentY} hasBall={spatial.available} frameId={spatial.frameId} analysisId={currentFrame?.analysis_id} trajectory={currentFrame?.intelligence?.trajectory} />
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
                      <span className="text-slate-200">Not measured</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Average Rally Speed:</span>
                      <span className="text-slate-200">Not measured</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Peak Recorded Velocity:</span>
                      <span className="text-amber-400 font-bold">{stats.maxVelocity.toFixed(1)} km/h</span>
                    </div>
                  </div>

                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-900 font-mono text-xs space-y-2">
                    <span className="text-emerald-400 font-bold block">SPIN FREQUENCY ANALYSIS</span>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Topspin Drives:</span>
                      <span className="text-slate-200">Not measured</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Backspin Chops / Pushes:</span>
                      <span className="text-slate-200">Not measured</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Sidespin Hook Serves:</span>
                      <span className="text-slate-200">Not measured</span>
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
      {!cinematicMode && <RallyTimeline data={currentFrame?.intelligence} canSeek={isSimulated} sendCommand={sendCommand}/>}

      {/* Architecture Pipeline Modal Overlay */}
      <AnimatePresence>
        {isModalOpen && <ArchitectureModal onClose={() => setIsModalOpen(false)} />}
      </AnimatePresence>
    </div>
  );
}
