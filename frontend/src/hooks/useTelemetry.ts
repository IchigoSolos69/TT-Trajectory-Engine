import { useState, useEffect, useCallback, useRef, RefObject } from 'react';
import { TelemetryFrame } from '../types/telemetry';
import { decodedFrames } from '../lib/decodedFrames';

export interface TelemetryStats {
  rallyStrokes: number;
  maxVelocity: number;
  elapsedTimeSec: number;
}

const emptyStats: TelemetryStats = { rallyStrokes: 0, maxVelocity: 0, elapsedTimeSec: 0 };

export function useTelemetry(_videoRef?: RefObject<HTMLVideoElement>, _targetFps = 60) {
  const [currentFrame, setCurrentFrame] = useState<TelemetryFrame | null>(null);
  const [isSimulated, setIsSimulated] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [stats, setStats] = useState<TelemetryStats>(emptyStats);
  const [impactPoints, setImpactPoints] = useState<Array<{ x: number; y: number }>>([]);
  const socketRef = useRef<WebSocket | null>(null);

  const sendCommand = useCallback((cmd: string) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) socketRef.current.send(cmd);
  }, []);
  const resetStats = useCallback(() => {
    setImpactPoints([]);
    setStats(emptyStats);
  }, []);

  useEffect(() => {
    let cancelled = false;
    let lastFrame = -1;
    let lastMode = "";
    let lastSource: number | undefined;
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;

    function disconnected() {
      setIsConnected(false);
      setIsSimulated(false);
      setCurrentFrame(null);
      setStats(emptyStats);
      setImpactPoints([]);
      if (!cancelled && reconnectTimer === undefined) {
        reconnectTimer = setTimeout(() => {
          reconnectTimer = undefined;
          connect();
        }, 3000);
      }
    }

    function connect() {
      if (cancelled) return;
      let ws: WebSocket;
      try {
        ws = new WebSocket('ws://localhost:9002');
      } catch (error) {
        console.error('[useTelemetry] Connection failed:', error);
        disconnected();
        return;
      }
      socketRef.current = ws;
      const active = () => !cancelled && socketRef.current === ws;
      ws.onopen = () => {
        if (active()) setIsConnected(true);
      };
      const publish = (data: TelemetryFrame) => {
          if (!active()) return;
          if (data.mode !== lastMode || data.source_id !== lastSource || data.frame_id < lastFrame) setImpactPoints([]);
          const fresh = data.mode !== lastMode || data.source_id !== lastSource || data.frame_id !== lastFrame;
          lastMode = data.mode ?? ''; lastFrame = data.frame_id; lastSource = data.source_id;
          setCurrentFrame(data);
          setIsSimulated(data.mode === 'simulation');
          if (data.impact_points) {
            setImpactPoints(data.impact_points.map(([x,y])=>({x,y})));
          } else setImpactPoints([]);
          setStats({
            rallyStrokes: data.rally?.stroke_count ?? 0,
            maxVelocity: data.rally?.max_speed_kmh ?? 0,
            elapsedTimeSec: data.rally?.rally_duration_s ?? 0,
          });
      };
      let decoding=false, pending: TelemetryFrame | null=null;
      const decode = () => {
        if(decoding || !pending || !active()) return;
        const data=pending;pending=null;
        if(!data.preview_jpeg){publish(data);return;}
        decoding=true;
        const image=new Image();
        image.onload=()=>{
          decoding=false;
          if(active() && (!pending || (pending.source_id===data.source_id && pending.mode===data.mode && pending.frame_id>=data.frame_id))) {
            decodedFrames.set(data,image);publish(data);
          }
          decode();
        };
        image.onerror=()=>{decoding=false;if(active()){console.error('[useTelemetry] Source image decode failed');setCurrentFrame(null);}decode();};
        image.src=`data:image/jpeg;base64,${data.preview_jpeg}`;
      };
      ws.onmessage = (event) => {
        if (!active()) return;
        try {
          const data: TelemetryFrame = JSON.parse(event.data);
          if (data.mode !== 'simulation' && data.mode !== 'live') throw new Error('Telemetry must declare its footage source.');
          pending=data;decode();
        } catch (error) {
          console.error('[useTelemetry] Invalid telemetry:', error);
          setCurrentFrame(null);
        }
      };
      ws.onerror = (error) => {
        if (!active()) return;
        console.error('[useTelemetry] WebSocket error:', error);
        ws.close();
      };
      ws.onclose = () => {
        if (!active()) return;
        socketRef.current = null;
        disconnected();
      };
    }
    connect();
    return () => {
      cancelled = true;
      if (reconnectTimer !== undefined) clearTimeout(reconnectTimer);
      const ws = socketRef.current;
      socketRef.current = null;
      if (ws) {
        ws.onopen = ws.onmessage = ws.onerror = ws.onclose = null;
        ws.close();
      }
    };
  }, []);

  return { currentFrame, isSimulated, isConnected, sendCommand, stats, resetStats, impactPoints };
}
