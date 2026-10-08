import type { TelemetryFrame } from '../types/telemetry';

// Both views consume the same observed pixel projected onto the calibrated table.
// Monocular height is unknown; never add a simulated flight to measured telemetry.
export function spatialFrame(frame: TelemetryFrame | null) {
  if(frame?.intelligence){
    const sample=frame.intelligence.trajectory.at(-1);
    const point=sample?.table_position;
    const available=!!point && point.every(Number.isFinite) && sample?.state!=='LOST';
    return {available,x:available?point![0]:.7625,y:available?point![1]:1.37,frameId:frame.frame_id};
  }
  const point = frame?.table_position;
  const available = !!frame?.ball && !!point && point.every(Number.isFinite);
  return { available, x: available ? point![0] : .7625, y: available ? point![1] : 1.37, frameId: frame?.frame_id ?? 0 };
}

export function tableToScene(x: number, y: number): [number, number, number] {
  return [x - .7625, .78, y - 1.37];
}
