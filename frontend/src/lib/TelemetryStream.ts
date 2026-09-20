import { TelemetryFrame, BallTelemetry } from '../types/telemetry';

export type TelemetryListener = (frame: TelemetryFrame) => void;

export class TelemetryStreamManager {
  private ws: WebSocket | null = null;
  private listeners: Set<TelemetryListener> = new Set();
  private isSimulating: boolean = false;
  private simInterval: NodeJS.Timeout | null = null;
  private url: string;

  // Simulation physics state
  private simFrameId = 0;
  private simTime = 0;
  private simBall = { x: 0.2, y: 0.7, z: 0.95, vx: 18.5, vy: -4.2, vz: 3.1 };

  constructor(url: string = 'ws://localhost:9002') {
    this.url = url;
  }

  public connect(): void {
    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log('[TelemetryStream] Connected to C++ uWebSockets backend:', this.url);
        this.stopSimulation();
      };

      this.ws.onmessage = (event) => {
        try {
          const data: TelemetryFrame = JSON.parse(event.data);
          this.notifyListeners(data);
        } catch (err) {
          console.error('[TelemetryStream] Failed to parse JSON message:', err);
        }
      };

      this.ws.onerror = (err) => {
        console.warn('[TelemetryStream] WebSocket error, starting fallback simulation mode.');
        this.startSimulation();
      };

      this.ws.onclose = () => {
        console.warn('[TelemetryStream] Connection closed, starting fallback simulation mode.');
        this.startSimulation();
      };
    } catch (e) {
      console.warn('[TelemetryStream] Could not initialize WebSocket, starting simulation mode.');
      this.startSimulation();
    }
  }

  public subscribe(listener: TelemetryListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notifyListeners(frame: TelemetryFrame): void {
    this.listeners.forEach((fn) => fn(frame));
  }

  public startSimulation(): void {
    if (this.isSimulating) return;
    this.isSimulating = true;
    console.log('[TelemetryStream] Active Mode: Simulated 120 FPS Table Tennis Telemetry');

    // Default table corners (normalized screen coords)
    const tableCorners: Array<[number, number]> = [
      [0.28, 0.52], // Top-Left
      [0.72, 0.52], // Top-Right
      [0.85, 0.88], // Bottom-Right
      [0.15, 0.88], // Bottom-Left
    ];

    let strokeCount = 12;
    let t = 0;
    const dt = 1 / 120; // 120 FPS telemetry loop

    this.simInterval = setInterval(() => {
      this.simFrameId++;
      t += dt;

      // Parabolic arc with bounce simulation for realistic table tennis telemetry
      const cycle = (t % 2.4) / 2.4; // 2.4 seconds per full serve/rally loop
      
      // Calculate smooth trajectory path
      const arcX = 0.5 + Math.sin(cycle * Math.PI * 2) * 0.28;
      const rawY = 0.45 + Math.abs(Math.sin(cycle * Math.PI * 3)) * 0.38;
      const rawZ = 0.76 + Math.abs(Math.sin(cycle * Math.PI * 4)) * 0.45;

      const isBounce = rawY > 0.81 && (cycle % 0.33 < 0.04);
      const speedKmh = 68.5 + Math.cos(cycle * Math.PI * 4) * 22.0;

      const ballData: BallTelemetry = {
        x: Number(arcX.toFixed(4)),
        y: Number(rawY.toFixed(4)),
        z: Number(rawZ.toFixed(3)),
        vx: Number((Math.cos(cycle * 10) * 12.5).toFixed(2)),
        vy: Number((-Math.sin(cycle * 10) * 15.0).toFixed(2)),
        vz: Number((Math.sin(cycle * 15) * 4.2).toFixed(2)),
        speed_kmh: Number(speedKmh.toFixed(1)),
        confidence: 0.96,
        is_bounce: isBounce,
        bounce_type: isBounce ? 'table_player2' : undefined,
      };

      const frame: TelemetryFrame = {
        frame_id: this.simFrameId,
        timestamp: Date.now(),
        fps: 120,
        ball: ballData,
        spin: {
          rpm: Math.round(4200 + Math.sin(cycle * 8) * 1800),
          type: cycle > 0.5 ? 'topspin' : 'sidespin_right',
          axis_deg: 45,
        },
        players: [
          {
            id: 1,
            label: 'Player A (Near)',
            bbox: [0.38, 0.78, 0.24, 0.20],
            confidence: 0.98,
          },
          {
            id: 2,
            label: 'Player B (Far)',
            bbox: [0.42, 0.36, 0.16, 0.18],
            confidence: 0.95,
          },
        ],
        table_corners: tableCorners,
        rally: {
          in_progress: true,
          stroke_count: strokeCount,
          rally_duration_s: Number(t.toFixed(1)),
          max_speed_kmh: 94.2,
        },
      };

      this.notifyListeners(frame);
    }, 1000 / 60); // Broadcast updates to UI at 60Hz
  }

  public stopSimulation(): void {
    if (this.simInterval) {
      clearInterval(this.simInterval);
      this.simInterval = null;
    }
    this.isSimulating = false;
  }

  public disconnect(): void {
    this.stopSimulation();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}
