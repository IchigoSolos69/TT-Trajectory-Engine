import { TelemetryFrame } from '../types/telemetry';

// Weak keys let old decoded images be collected as soon as views release frames.
export const decodedFrames = new WeakMap<TelemetryFrame, HTMLImageElement>();
