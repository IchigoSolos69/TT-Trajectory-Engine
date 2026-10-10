export type SpinType = 'topspin' | 'backspin' | 'sidespin_left' | 'sidespin_right' | 'flat';

export interface BallTelemetry {
  x: number;             // Normalized 0..1 x-coordinate
  y: number;             // Normalized 0..1 y-coordinate
  z: number | null;      // Unknown until physical height is supported by reconstruction.
  vx: number;            // Image-space X velocity (normalized units/s)
  vy: number;            // Image-space Y velocity (normalized units/s)
  vz: number | null;     // Unobserved without supported physical reconstruction.
  speed_kmh: number;     // Absolute speed in km/h
  confidence: number;    // Model score 0..1; see ball_model.score_kind.
  is_bounce: boolean;
  bounce_x?: number;
  bounce_y?: number;
  bounce_type?: 'table_player1' | 'table_player2' | 'net' | 'out' | 'paddle';
}

export interface PlayerDetection {
  id: number;
  label: 'Player A (Near)' | 'Player B (Far)';
  bbox: [number, number, number, number]; // [x, y, w, h] normalized 0..1
  confidence: number;
}

export interface TelemetryFrame {
  ball_flight?: {status:'EXPERIMENTAL_FORECAST'|'UNAVAILABLE';reason:string;method:string;coordinate_basis:'source_pixels';reference_frame:number;reference_timestamp:number;segment_id:number;support_frames:number[];maximum_horizon_s:number;fit_rmse_px:number|null;uncertainty:null;uncertainty_reason:string;points:Array<{timestamp:number;pixel:[number,number]}>};
  ball_model?: {backend: 'yolo'|'blurball';model_sha256: string;experimental: boolean;score_kind: string;score_calibrated: boolean;temporal_policy: string;lookahead_frames: number;warmup_frames: number;heatmap_threshold: number|null;timestamp_basis: string};
  ball_blur?: {center_px:[number,number];endpoints_px:[[number,number],[number,number]];extent_px:number;axis_deg:number;component_mass:number;heatmap_peak:number;axis_directed:false;exposure_seconds:null;gpu_ms:number}|null;
  schema_version?: number;
  analysis_id?: string;
  presentation_epoch?: number;
  analysis?: {status: 'RUNNING'|'COMPLETE'|'PARTIAL'|'ERROR'; analyzed_frames: number; total_frames: number; waiting_for_analysis: boolean; requested_frame: number; coverage: string};
  intelligence?: IntelligenceTelemetry;
  source_id?: number;
  height_observed?: boolean;
  position_basis?: string;
  video_picker_open?: boolean;
  video_error?: string;
  strikes?: Array<{id: number; frame_id: number; source_time: number; speed_kmh: number}>;
  preview_jpeg?: string;
  source_width?: number;
  source_height?: number;
  source_time?: number;
  total_frames?: number;
  playback_rate?: number;
  processing_fps?: number;
  preview_fps?: number;
  skipped_frames?: number;
  camera_active?: boolean;
  camera_index?: number;
  training_recording?: { active: boolean; failed: boolean; saved_frames: number; dropped_frames: number; clip: string };
  impact_points?: Array<[number, number]>;
  image_impact_points?: Array<[number, number]>;
  paused?: boolean;
  ended?: boolean;
  table_position?: [number, number];
  table_calibration_status?: string;
  scene_detections?: Array<{ class_id: number; confidence: number; bbox: [number, number, number, number] }>;
  scene_frame_id?: number;
  suppressed_scene_detections?: Array<{class_id: number; confidence: number; reason: string; bbox: [number,number,number,number]}>;
  ball_observation?: {state: 'DETECTOR_ONLY'|'MEASURED'|'PREDICTED'|'REACQUIRED'|'REJECTED'|'LOST';reason: string};
  source_video?: string;
  mode?: 'live' | 'simulation' | string;
  bounce_detector_status?: string;
  x?: number;
  y?: number;
  z?: number | null;
  velocity?: number;
  bounce?: boolean;
  bounce_x?: number;
  bounce_y?: number;
  latency_ms?: number;
  infer_ms?: number;
  frame_id: number;
  timestamp: number;     // Microseconds or ms
  fps: number;
  ball: BallTelemetry | null;
  spin: {
    rpm: number | null;
    type: SpinType | 'camera_axis';
    axis_deg: number | null;
    status?: 'estimated' | 'unavailable';
    quality?: number;
    reason?: string;
  };
  players: PlayerDetection[];
  table_corners: Array<[number, number]>; // 4 points defining table top surface
  rally: {
    in_progress: boolean;
    stroke_count: number;
    rally_duration_s: number;
    max_speed_kmh: number;
  };
}

export interface TrajectorySample {
  frame_id: number; timestamp: number; segment_id: number;
  state: 'MEASURED' | 'PREDICTED' | 'REACQUIRED' | 'REJECTED' | 'LOST';
  reason: string; accepted: boolean; pixel: [number,number]; raw_observation: [number,number] | null;
  table_position: [number,number] | null; z: null; velocity: [number,number] | null;
  covariance: [number,number,number,number] | null; ellipse: [number,number,number] | null;
  detector_score: number; tracking_quality: number; calibration_quality: number;
  blur_score: number; contrast: number; elongation: number; blur_angle_rad: number;
  occlusion_evidence: number; mahalanobis_squared: number; missed_frames: number;
}
export interface IntelligenceTelemetry {
  rally_graph?: {status: string;events: RallyEvent[];highlights: Array<{event_id:number;start_s:number;end_s:number;reason:string}>};
  schema_version: number;
  provenance: {position: string; height: string; covariance: string; quality: string; ball_model_sha256: string; scene_model_sha256: string};
  calibration: {state: string; usable: boolean; quality: number; corner_residual_px: number; corner_sigma_px: number; border_support: number; age_s: number; epoch: number; net_landmark: string; center_line_landmark: string; reprojection_error_px: number|null};
  trajectory: TrajectorySample[];
}
export interface RallyEvent {
  id:number;previous_event_id:number;rally_id:number;frame_id:number;timestamp:number;type:string;status:string;reason:string;
  position:[number,number]|null;incoming_speed_kmh:number;outgoing_speed_kmh:number;heading_change_deg:number;
  acceleration_estimate_m_s2:number;fit_residual_m:number;quality:number;player_side:number|null;flight_duration_s:number;
}

export interface PhysicsParams {
  speed_kmh: number;
  launch_angle_deg: number;
  side_angle_deg: number;
  spin_rpm: number;
  spin_type: SpinType;
  drag_coefficient: number; // Air resistance Cd ~0.40 for 40mm ball
  table_restitution: number; // Coefficient of restitution e ~0.89
  magnus_coefficient: number; // Magnus force intensity multiplier
  table_friction: number; // Friction during bounce ~0.25
}
