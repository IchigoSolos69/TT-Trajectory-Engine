'use client';
import type { IntelligenceTelemetry } from '../types/telemetry';
export function MeasurementQuality({data}:{data?:IntelligenceTelemetry}){
  if(!data)return null;
  const sample=data.trajectory.at(-1),c=data.calibration;
  return <section aria-label="Measurement provenance" className="rounded-xl border border-cyan-900/60 bg-slate-950 p-4 text-xs font-mono space-y-3">
    <div className="flex flex-wrap gap-x-6 gap-y-2">
      <span className="text-cyan-300">TRAJECTORY · {sample?.state??'LOST'}</span>
      <span className="text-emerald-300">TABLE · {c.state}</span>
      <span className="text-slate-300">Geometry quality {(c.quality*100).toFixed(0)}/100</span>
      <span className="text-slate-400">Corner stability residual {c.corner_residual_px.toFixed(1)} px · age {c.age_s.toFixed(2)} s</span>
    </div>
    {sample && <div className="flex flex-wrap gap-x-6 gap-y-2 text-slate-400">
      <span>Gate: {sample.reason.replaceAll('_',' ')}</span>
      <span>Blur score {sample.blur_score.toFixed(2)} · misses {sample.missed_frames}</span>
      <span>Innovation D² {sample.mahalanobis_squared.toFixed(1)}</span>
      {sample.ellipse&&<span className="text-amber-300">Model ellipse ±{(sample.ellipse[0]*100).toFixed(1)} / ±{(sample.ellipse[1]*100).toFixed(1)} cm</span>}
    </div>}
    <p className="text-slate-500">Solid: observed · dashed: predicted · green: reacquired · red: rejected observation. Bands use model covariance; 95% coverage has not been validated. Quality scores are not probabilities. Height, net landmarks and spin axis remain unobserved.</p>
    <details className="text-slate-500"><summary className="cursor-pointer">Model provenance</summary><p className="break-all mt-2">Ball SHA-256: {data.provenance.ball_model_sha256}<br/>Scene SHA-256: {data.provenance.scene_model_sha256}</p></details>
  </section>;
}
