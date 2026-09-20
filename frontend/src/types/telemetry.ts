export type SpinType = 'topspin' | 'backspin' | 'sidespin_left' | 'sidespin_right' | 'flat';

export interface BallTelemetry {
  x: number;             // Normalized 0..1 x-coordinate
  y: number;             // Normalized 0..1 y-coordinate
  z: number;             // Estimated 3D elevation in meters (table height = 0.76m)
  vx: number;            // Velocity X component (m/s)
  vy: number;            // Velocity Y component (m/s)
  vz: number;            // Velocity Z component (m/s)
  speed_kmh: number;     // Absolute speed in km/h
  confidence: number;    // Detection confidence 0..1
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
  mode?: 'live' | 'simulation' | string;
  x?: number;
  y?: number;
  z?: number;
  velocity?: number;
  bounce?: boolean;
  bounce_x?: number;
  bounce_y?: number;
  frame_id: number;
  timestamp: number;     // Microseconds or ms
  fps: number;
  ball: BallTelemetry | null;
  spin: {
    rpm: number;
    type: SpinType;
    axis_deg: number;
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
