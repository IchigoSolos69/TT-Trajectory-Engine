import { useState, useEffect, useCallback, useRef, RefObject } from 'react';
import { TelemetryFrame } from '../types/telemetry';

export interface TelemetryStats {
  rallyStrokes: number;
  maxVelocity: number;
  elapsedTimeSec: number;
}

export function useTelemetry(videoRef?: RefObject<HTMLVideoElement>, targetFps: number = 60) {
  const [currentFrame, setCurrentFrame] = useState<TelemetryFrame | null>(null);
  const [isSimulated, setIsSimulated] = useState<boolean>(false);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [socket, setSocket] = useState<WebSocket | null>(null);

  // Real-time aggregated statistics from C++ physics payload
  const [stats, setStats] = useState<TelemetryStats>({
    rallyStrokes: 0,
    maxVelocity: 0,
    elapsedTimeSec: 0,
  });

  const [impactPoints, setImpactPoints] = useState<Array<{ x: number; y: number }>>([]);

  const frameCount = useRef(0);

  const sendCommand = useCallback((cmd: string) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      console.log(`[useTelemetry] Transmitting command to C++ Backend: ${cmd}`);
      socket.send(cmd);
    }
  }, [socket]);

  const resetStats = useCallback(() => {
    frameCount.current = 0;
    setImpactPoints([]);
    setStats({ rallyStrokes: 0, maxVelocity: 0, elapsedTimeSec: 0 });
  }, []);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let animFrameId: number;
    let mockFrames: TelemetryFrame[] = [];
    let isCancelled = false;

    async function initFallback() {
      if (isCancelled) return;
      setIsSimulated(true);
      setIsConnected(false);

      try {
        const res = await fetch('/data/mock_telemetry.json');
        if (res.ok) {
          mockFrames = await res.json();
        }
      } catch (e) {
        console.warn('[useTelemetry] Failed to load mock_telemetry.json:', e);
      }

      // Loop sync function
      const syncLoop = () => {
        if (isCancelled) return;

        if (videoRef?.current && mockFrames.length > 0) {
          const frameIdx = Math.floor(videoRef.current.currentTime * targetFps) % mockFrames.length;
          if (mockFrames[frameIdx]) {
            setCurrentFrame(mockFrames[frameIdx]);
          }
        } else if (mockFrames.length > 0) {
          const idx = Math.floor(Date.now() / (1000 / targetFps)) % mockFrames.length;
          const frame = mockFrames[idx];
          setCurrentFrame(frame);
          if (frame) {
            frameCount.current += 1;
            const velocity = frame.ball?.speed_kmh ?? frame.velocity ?? 0;
            const isBounce = frame.ball?.is_bounce ?? frame.bounce ?? false;
            if (isBounce) {
              const bx = frame.bounce_x ?? frame.ball?.bounce_x ?? frame.ball?.x ?? 0.5;
              const by = frame.bounce_y ?? frame.ball?.bounce_y ?? frame.ball?.y ?? 0.5;
              setImpactPoints(prev => [...prev.slice(-49), { x: bx, y: by }]);
            }
            setStats((prev) => ({
              rallyStrokes: isBounce ? prev.rallyStrokes + 1 : prev.rallyStrokes,
              maxVelocity: Math.max(prev.maxVelocity, velocity),
              elapsedTimeSec: frameCount.current / targetFps,
            }));
          }
        }

        animFrameId = requestAnimationFrame(syncLoop);
      };

      animFrameId = requestAnimationFrame(syncLoop);
    }

    try {
      ws = new WebSocket('ws://localhost:9002');
      setSocket(ws);

      ws.onopen = () => {
        console.log('[useTelemetry] Connected to C++ uWebSockets Engine on port 9002');
        setIsConnected(true);
        setIsSimulated(false);
      };

      ws.onmessage = (event) => {
        try {
          const data: TelemetryFrame = JSON.parse(event.data);
          setCurrentFrame(data);
          if (data.mode) {
            setIsSimulated(data.mode === 'simulation');
          }

          frameCount.current += 1;
          const currentVelocity = data.ball?.speed_kmh ?? data.velocity ?? 0;
          const isStrike = (data as any).is_strike ?? (data as any).strike ?? false;
          const isBounce = (data as any).is_bounce ?? (data as any).bounce ?? false;

          if (isBounce) {
            const bx = (data as any).bounce_x ?? data.ball?.bounce_x ?? data.ball?.x ?? 0.5;
            const by = (data as any).bounce_y ?? data.ball?.bounce_y ?? data.ball?.y ?? 0.5;
            setImpactPoints(prev => [...prev.slice(-49), { x: bx, y: by }]);
          }

          const serverStrokes = data.rally?.stroke_count ?? (data as any).stroke_count ?? (data as any).rally_strokes;
          const serverMaxVelocity = data.rally?.max_speed_kmh ?? (data as any).max_velocity ?? (data as any).max_speed_kmh;
          const serverElapsedTime = data.rally?.rally_duration_s ?? (data as any).elapsed_time;

          setStats((prev) => ({
            rallyStrokes: typeof serverStrokes === 'number' ? serverStrokes : (isStrike ? prev.rallyStrokes + 1 : prev.rallyStrokes),
            maxVelocity: typeof serverMaxVelocity === 'number' ? serverMaxVelocity : Math.max(prev.maxVelocity, currentVelocity),
            elapsedTimeSec: typeof serverElapsedTime === 'number' ? serverElapsedTime : (frameCount.current / 120),
          }));
        } catch (err) {
          console.error('[useTelemetry] JSON parse error:', err);
        }
      };

      ws.onerror = () => {
        initFallback();
      };

      ws.onclose = () => {
        initFallback();
      };
    } catch (e) {
      initFallback();
    }

    return () => {
      isCancelled = true;
      if (ws) {
        ws.onopen = null;
        ws.onmessage = null;
        ws.onerror = null;
        ws.onclose = null;
        if (ws.readyState === WebSocket.OPEN) {
          ws.close();
        }
      }
      if (animFrameId) cancelAnimationFrame(animFrameId);
    };
  }, [videoRef, targetFps]);

  return { currentFrame, isSimulated, isConnected, sendCommand, stats, resetStats, impactPoints };
}
