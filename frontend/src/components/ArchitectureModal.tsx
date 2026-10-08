'use client';

import React from 'react';
import { motion } from 'framer-motion';

interface ArchitectureModalProps {
  onClose: () => void;
}

export const ArchitectureModal: React.FC<ArchitectureModalProps> = ({ onClose }) => {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm">
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 20 }}
        transition={{ duration: 0.2, ease: 'easeOut' }}
        className="relative w-full max-w-4xl bg-slate-900 border border-slate-700 rounded-xl shadow-2xl overflow-hidden"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-800 bg-slate-900/50">
          <h2 className="text-xl font-bold tracking-widest text-emerald-400 uppercase">
            System Architecture & Pipeline
          </h2>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white transition-colors p-2 text-lg font-bold"
          >
            ✕
          </button>
        </div>

        {/* Pipeline Diagram Grid */}
        <div className="p-8 space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {/* Step 1: Ingestion */}
            <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700 relative">
              <div className="text-xs font-mono text-slate-500 mb-2">STAGE 01</div>
              <h3 className="font-bold text-slate-200 mb-1">C++ Video Ingestion</h3>
              <p className="text-sm text-slate-400">
                OpenCV reads the selected camera or dataset file. Letterboxing preserves image geometry before transferring the frame to the GPU.
              </p>
              {/* Arrow linking to next step */}
              <div className="hidden md:block absolute -right-3 top-1/2 -translate-y-1/2 text-slate-600 font-bold">
                →
              </div>
            </div>

            {/* Step 2: Inference */}
            <div className="p-4 rounded-lg bg-emerald-900/20 border border-emerald-800/50 relative shadow-[0_0_15px_rgba(16,185,129,0.1)]">
              <div className="text-xs font-mono text-emerald-500 mb-2">STAGE 02</div>
              <h3 className="font-bold text-emerald-300 mb-1">TensorRT Engine</h3>
              <p className="text-sm text-slate-400">
                Separate YOLO11 ball and scene detectors execute in TensorRT using FP16 computation and{' '}
                <code className="text-emerald-400 bg-emerald-950 px-1 rounded font-mono">
                  cudaStreamAsync
                </code>{' '}
                with runtime timing reported in the engine view.
              </p>
              <div className="hidden md:block absolute -right-3 top-1/2 -translate-y-1/2 text-emerald-800 font-bold">
                →
              </div>
            </div>

            {/* Step 3: Tracking & Physics */}
            <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700 relative">
              <div className="text-xs font-mono text-slate-500 mb-2">STAGE 03</div>
              <h3 className="font-bold text-slate-200 mb-1">Tracker & Projection</h3>
              <p className="text-sm text-slate-400">
                ByteTrack-style association uses constant-velocity prediction and two-stage IoU matching without ReID. Detected table bounds support approximate planar projection. Ball height and spin remain model estimates.
              </p>
              <div className="hidden md:block absolute -right-3 top-1/2 -translate-y-1/2 text-slate-600 font-bold">
                →
              </div>
            </div>

            {/* Step 4: Telemetry Bridge */}
            <div className="p-4 rounded-lg bg-blue-900/20 border border-blue-800/50 shadow-[0_0_15px_rgba(59,130,246,0.1)]">
              <div className="text-xs font-mono text-blue-500 mb-2">STAGE 04</div>
              <h3 className="font-bold text-blue-300 mb-1">WebSocket Streaming</h3>
              <p className="text-sm text-slate-400">
                <code className="text-blue-400 bg-blue-950 px-1 rounded font-mono">
                  Non-blocking sockets
                </code>{' '}
                deliver matching imagery and detections together. A bounded queue keeps the latest waiting frame; preview delivery is capped near 30 FPS.
              </p>
            </div>
          </div>

          {/* GitHub / Repo CTA */}
          <div className="mt-8 p-4 bg-slate-950 rounded-lg border border-slate-800 flex flex-col sm:flex-row justify-between items-center gap-3">
            <span className="text-sm text-slate-400 font-mono">
              View the bare-metal C++ source code and CMake configurations:
            </span>
            <a
              href="https://github.com/IchigoSolos69"
              target="_blank"
              rel="noopener noreferrer"
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white text-sm font-bold rounded-lg transition-colors whitespace-nowrap"
            >
              View on GitHub ↗
            </a>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default ArchitectureModal;
