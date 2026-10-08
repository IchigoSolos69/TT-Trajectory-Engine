'use client';
import { useState } from 'react';

const preparation = [
  { id: 'view', title: 'Frame the whole court', text: 'Mount the camera firmly and clean the lens. Keep all four table corners, the net and both players in view, with space around the table for the ball’s flight. Keep the camera fixed during each clip.' },
  { id: 'image', title: 'Check the ball at both ends', text: 'Use bright, even light; avoid glare and a bright window behind the players. Check near and far positions. If your camera supports it, adjust focus and shorten exposure until the moving ball is distinguishable; add light if the image becomes too dark.' },
  { id: 'sample', title: 'Review a short trial first', text: 'Authorize the chosen camera, record a short trial, stop and open Review saved frames. Inspect several clean images at their original size for ball visibility, cropping and motion blur. Adjust the setup before recording more.' },
  { id: 'sessions', title: 'Reserve separate evaluation sessions', text: 'Plan separate training, validation and test sessions before collecting. Keep every clip from one session in the same split. Use later sessions with different lighting or camera positions to check generalization.' },
];

export function TrainingRecordingGuide() {
  const [checked, setChecked] = useState<string[]>([]);
  return <section aria-label="Live training recording guide" className="space-y-4">
    <div className="flex flex-wrap items-center justify-between gap-2">
      <div>
        <h3 className="font-semibold text-slate-100">Make each recording useful for training</h3>
        <p className="mt-1 text-xs text-slate-400">Follow this routine for the actual gameplay camera. Repeat it for room examples when adapting detection.</p>
      </div>
      <span className="text-xs font-mono text-cyan-300" role="status">{checked.length} / {preparation.length} setup reminders checked</span>
    </div>
    <div className="grid gap-3 md:grid-cols-2">
      {preparation.map((step, index) => <label key={step.id} className={`flex gap-3 rounded-xl border p-3 cursor-pointer ${checked.includes(step.id) ? 'border-emerald-500/40 bg-emerald-500/5' : 'border-slate-700/70 bg-slate-950/40'}`}>
        <input type="checkbox" className="mt-1 shrink-0 accent-emerald-400" checked={checked.includes(step.id)} onChange={e => setChecked(values => e.target.checked ? [...values, step.id] : values.filter(id => id !== step.id))} />
        <span><span className="block text-sm font-semibold text-slate-200">{index + 1}. {step.title}</span><span className="mt-1 block text-xs leading-relaxed text-slate-400">{step.text}</span></span>
      </label>)}
    </div>
    <p className="text-xs text-slate-500">These checks are your observations, not an automatic image-quality assessment. They do not block recording.</p>
    <details className="rounded-xl border border-slate-700/70 bg-slate-950/30 p-3" open>
      <summary className="cursor-pointer text-sm font-semibold text-cyan-300">What to capture — follow this order</summary>
      <ol className="mt-3 list-decimal pl-5 space-y-3 text-xs leading-relaxed text-slate-300">
        <li><strong className="text-slate-100">Record verified absence.</strong> Remove the ball from view and record the table with players, then an empty scene. Include ordinary furniture and white objects that could cause false detections. A ball hidden behind a player is occluded, not a verified absent-ball example.</li>
        <li><strong className="text-slate-100">Record clear ball examples.</strong> Show the ball near and far from the camera, stationary and moving. Include both table ends and different background regions. Verify it can be located in the saved images.</li>
        <li><strong className="text-slate-100">Record normal gameplay.</strong> Capture serves and rallies from both players, short and deep placements, and left/right directions. Stop and repeat for another clip when the 60-second capture ends.</li>
        <li><strong className="text-slate-100">Repeat with temporal bursts.</strong> Use Record temporal bursts for moving balls and brief paddle/player occlusions. Bursts are taken once per second and can miss a contact; repeat across several rallies so you capture varied sequences.</li>
        <li><strong className="text-slate-100">Add difficult cases alongside clear ones.</strong> Include faster shots, natural blur, brief occlusions and lighting changes. Change the camera angle between clips, then hold it steady. Capture difficult examples can supplement these clips when available.</li>
        <li><strong className="text-slate-100">Check the saved result.</strong> Stop recording, inspect frames and any dropped-image warning, then repeat interrupted clips. Collect another session later; repeated copies of the same rally add little new information.</li>
      </ol>
    </details>
    <details className="rounded-xl border border-slate-700/70 bg-slate-950/30 p-3">
      <summary className="cursor-pointer text-sm font-semibold text-emerald-300">After recording — review before training</summary>
      <ol className="mt-3 list-decimal pl-5 space-y-2 text-xs leading-relaxed text-slate-300">
        <li>Open <strong>Review saved frames</strong>. Correct every visible ball, person and table box; remove false person boxes on furniture. A desk is not a table-tennis table.</li>
        <li>Tag ball visibility and blur. Add visible table corners, center-line endpoints and net endpoints. Leave hidden landmarks unknown. Only add an event when its contact is visible at that captured frame.</li>
        <li>Save reviewed labels only when you can label the frame reliably. An empty class list means you checked and verified absence. Exclude ambiguous or fully occluded frames rather than turning an unknown ball position into a negative label.</li>
        <li>Assign whole sessions to training, validation or testing. Keep evaluation sessions held out and remove duplicate images across splits. Audit complete temporal bursts before temporal training.</li>
        <li>Compare the candidate with the deployed model on the same held-out recordings, including ball-absent, blur and occlusion cases. More saved frames or higher model scores alone do not demonstrate better accuracy.</li>
      </ol>
    </details>
    <aside aria-label="Training recording warnings" className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-3">
      <h4 className="text-sm font-semibold text-amber-300">Recording limits to account for</h4>
      <ul className="mt-2 list-disc pl-4 space-y-2 text-xs leading-relaxed text-amber-100/80">
        <li><strong>Sampled images, not continuous match video.</strong> Keep a separate original video if you need full rally reconstruction, precise bounce timing or highlight clips. These controls save clean images without dashboard overlays.</li>
        <li><strong>Camera settings need your check.</strong> The app does not set or verify camera FPS, resolution, focus or exposure. Advertised 120 FPS is not proof of captured frame rate. Burst continuity refers to successful camera reads, and saved times are host capture-loop times.</li>
        <li><strong>Recording ends after 60 seconds</strong>, when the camera stops or switches, or when all dashboard clients disconnect. Check that frames were actually saved before starting the next clip.</li>
        <li><strong>Never fill a missing burst frame with a copy or interpolation.</strong> Incomplete or nonconsecutive five-frame bursts must be excluded from temporal training.</li>
        <li><strong>Suggestions need human review.</strong> Low confidence is not proof of an error; high confidence is not ground truth. Include representative clear footage as well as suspected failures.</li>
      </ul>
    </aside>
  </section>;
}
