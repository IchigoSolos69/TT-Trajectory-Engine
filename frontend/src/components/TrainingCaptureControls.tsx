'use client';
import { TelemetryFrame } from '../types/telemetry';
import { TrainingRecordingGuide } from './TrainingRecordingGuide';
import { TrainingSessionReadiness } from './TrainingSessionReadiness';

export function TrainingCaptureControls({ frame, sendCommand }: {
  frame: TelemetryFrame; sendCommand: (command: string) => void;
}) {
  const recording = frame.training_recording;
  return <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4 space-y-4" aria-label="Training capture controls">
    <div className="flex flex-wrap items-center gap-3 text-xs font-mono">
      <button disabled={!frame.camera_active}
        className={`px-4 py-2 rounded-lg font-bold disabled:opacity-40 ${recording?.active ? 'bg-red-500 text-white' : 'bg-emerald-400 text-slate-950'}`}
        onClick={() => sendCommand(recording?.active ? 'STOP_TRAINING_RECORDING' : 'START_TRAINING_RECORDING')}>
        {recording?.active ? 'Stop recording' : 'Record training clip'}
      </button>
      {!recording?.active && <button disabled={!frame.camera_active||!frame.intelligence} className="px-3 py-2 rounded-lg border border-amber-500/50 text-amber-300 disabled:opacity-40" onClick={()=>sendCommand('START_ACTIVE_LEARNING')}>Capture difficult examples</button>}
      {!recording?.active && <button disabled={!frame.camera_active} className="px-3 py-2 rounded-lg border border-cyan-500/50 text-cyan-300 disabled:opacity-40" onClick={()=>sendCommand('START_TEMPORAL_RECORDING')}>Record temporal bursts</button>}
      <a href="/training-review" className="text-cyan-300 underline">Review saved frames</a>
      <span className={recording?.active ? 'text-red-300' : 'text-slate-400'} role="status">
        {recording?.active ? 'RECORDING · ' : ''}{recording?.saved_frames ?? 0} clean frames saved
        {!!recording?.dropped_frames && ` · ${recording.dropped_frames} images not saved (capture queue full)`}
      </span>
    </div>
    <div className="grid gap-2 text-xs md:grid-cols-3" aria-label="Training capture mode guide">
      <div className="rounded-lg border border-emerald-500/25 bg-emerald-500/5 p-3"><p className="font-semibold text-emerald-300">Record training clip</p><p className="mt-1 leading-relaxed text-slate-400">Up to 5 clean images/second for ball, person and table detection. Use this for representative positive and verified absent examples.</p></div>
      <div className="rounded-lg border border-cyan-500/25 bg-cyan-500/5 p-3"><p className="font-semibold text-cyan-300">Record temporal bursts</p><p className="mt-1 leading-relaxed text-slate-400">5 consecutive captured frames once per second for motion and recovery training. Review complete bursts; this is not continuous video.</p></div>
      <div className="rounded-lg border border-amber-500/25 bg-amber-500/5 p-3"><p className="font-semibold text-amber-300">Capture difficult examples</p><p className="mt-1 leading-relaxed text-slate-400">Selects suspected failures at up to 5 images/second. Supplement normal footage; review each suggestion. Requires experimental intelligence.</p></div>
    </div>
    {!frame.camera_active && <p className="text-xs text-cyan-300">Authorize the chosen live camera above, inspect the preview, then choose a recording mode.</p>}
    <p className="text-xs text-slate-400">All modes save locally for at most 60 seconds. Nothing is uploaded.</p>
    {recording?.clip && <p className="text-xs font-mono text-emerald-300 break-all">Saved in training_data/live/{recording.clip}</p>}
    {recording?.failed && <p role="alert" className="text-xs text-red-300">Recording failed. Check the backend log and available disk space; saved frames remain in the clip folder.</p>}
    {!!recording?.dropped_frames && <p role="alert" className="text-xs text-amber-300">Some images were not saved. Check disk space and the saved sequence; exclude incomplete temporal bursts and repeat the affected capture.</p>}
    <p className="text-xs text-slate-400">Keep every clip from one session in the same training, validation or test group. Review labels before training.</p>
    <TrainingSessionReadiness />
    <p className="rounded-lg border border-amber-500/25 bg-amber-500/5 px-3 py-2 text-xs leading-relaxed text-amber-200">Before recording: keep the camera steady, frame the full table and both players, and inspect saved near/far ball images for visibility and blur. Recording automatically stops after 60 seconds or when the camera closes or switches.</p>
    <details className="rounded-xl border border-slate-700/70 bg-slate-950/30 p-3">
      <summary className="cursor-pointer text-sm font-semibold text-cyan-300">Recording checklist, gameplay steps & warnings</summary>
      <div className="mt-4"><TrainingRecordingGuide key={frame.source_id ?? frame.camera_index ?? 0} /></div>
    </details>
  </div>;
}
