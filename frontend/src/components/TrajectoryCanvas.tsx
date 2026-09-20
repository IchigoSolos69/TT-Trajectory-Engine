'use client';

import React, { useRef, useEffect, useState } from 'react';
import { TelemetryFrame, BallTelemetry } from '../types/telemetry';

interface TrajectoryCanvasProps {
  currentFrame: TelemetryFrame | null;
  historyLength?: number;
  showPlayerBoxes?: boolean;
  showTableCorners?: boolean;
  showVelocityVector?: boolean;
  showHeatmap?: boolean;
}

interface BounceImpact {
  x: number;
  y: number;
  alpha: number;
  radius: number;
}

export const TrajectoryCanvas: React.FC<TrajectoryCanvasProps> = ({
  currentFrame,
  historyLength = 40,
  showPlayerBoxes = true,
  showTableCorners = true,
  showVelocityVector = true,
  showHeatmap = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const historyRef = useRef<Array<{ x: number; y: number; timestamp: number; speed: number }>>([]);
  const bouncesRef = useRef<BounceImpact[]>([]);
  const heatmapRef = useRef<Array<{ x: number; y: number }>>([]);

  // Store telemetry history and process bounce ripples
  useEffect(() => {
    if (!currentFrame || !currentFrame.ball) return;

    const ball = currentFrame.ball;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const px = ball.x * canvas.width;
    const py = ball.y * canvas.height;

    // Append to trajectory tail
    historyRef.current.push({
      x: px,
      y: py,
      timestamp: currentFrame.timestamp,
      speed: ball.speed_kmh,
    });

    if (historyRef.current.length > historyLength) {
      historyRef.current.shift();
    }

    // Trigger bounce ripple animation
    if (ball.is_bounce) {
      bouncesRef.current.push({
        x: px,
        y: py,
        alpha: 1.0,
        radius: 4,
      });
      heatmapRef.current.push({ x: px, y: py });
    }
  }, [currentFrame, historyLength]);

  // Main Render Loop
  useEffect(() => {
    let animId: number;

    const render = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Handle retina canvas resolution scaling
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      ctx.clearRect(0, 0, width, height);

      // 1. Draw Table Tennis Court Perspective Grid
      drawTableGrid(ctx, width, height, currentFrame?.table_corners);

      // 2. Draw Impact Heatmap if enabled
      if (showHeatmap && heatmapRef.current.length > 0) {
        drawHeatmap(ctx, heatmapRef.current);
      }

      // 3. Draw Player Bounding Boxes with Tactical HUD style
      if (showPlayerBoxes && currentFrame?.players) {
        drawPlayerHUD(ctx, currentFrame.players, width, height);
      }

      // 4. Draw Trajectory Glow Tail
      drawTrajectoryTail(ctx, historyRef.current);

      // 5. Draw Active Ball & Velocity Vector
      if (currentFrame?.ball) {
        drawBallAndVector(ctx, currentFrame.ball, width, height, showVelocityVector);
      }

      // 6. Animate & Draw Bounce Ripples
      drawBounceRipples(ctx);

      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, [currentFrame, showPlayerBoxes, showTableCorners, showVelocityVector, showHeatmap]);

  // Canvas Drawing Helpers
  const drawTableGrid = (
    ctx: CanvasRenderingContext2D,
    w: number,
    h: number,
    corners?: Array<[number, number]>
  ) => {
    ctx.save();
    
    // Default perspective corners if not provided
    const pts = corners && corners.length === 4
      ? corners.map(([cx, cy]) => ({ x: cx * w, y: cy * h }))
      : [
          { x: w * 0.28, y: h * 0.52 },
          { x: w * 0.72, y: h * 0.52 },
          { x: w * 0.88, y: h * 0.88 },
          { x: w * 0.12, y: h * 0.88 },
        ];

    // Table Surface Fill
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    ctx.lineTo(pts[1].x, pts[1].y);
    ctx.lineTo(pts[2].x, pts[2].y);
    ctx.lineTo(pts[3].x, pts[3].y);
    ctx.closePath();

    const tableGrad = ctx.createLinearGradient(0, pts[0].y, 0, pts[2].y);
    tableGrad.addColorStop(0, 'rgba(11, 83, 69, 0.45)');
    tableGrad.addColorStop(1, 'rgba(27, 79, 114, 0.65)');
    ctx.fillStyle = tableGrad;
    ctx.fill();

    // Table White Borders
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 3;
    ctx.shadowColor = '#00f3ff';
    ctx.shadowBlur = 10;
    ctx.stroke();

    // Net Line (Center Division)
    const netLeftX = pts[0].x + (pts[3].x - pts[0].x) * 0.5;
    const netLeftY = pts[0].y + (pts[3].y - pts[0].y) * 0.5;
    const netRightX = pts[1].x + (pts[2].x - pts[1].x) * 0.5;
    const netRightY = pts[1].y + (pts[2].y - pts[1].y) * 0.5;

    ctx.beginPath();
    ctx.moveTo(netLeftX, netLeftY);
    ctx.lineTo(netRightX, netRightY);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.restore();
  };

  const drawPlayerHUD = (
    ctx: CanvasRenderingContext2D,
    players: Array<any>,
    w: number,
    h: number
  ) => {
    ctx.save();
    players.forEach((p) => {
      const [bx, by, bw, bh] = p.bbox;
      const x = bx * w;
      const y = by * h;
      const width = bw * w;
      const height = bh * h;

      const isNear = p.label.includes('Near');
      const accentColor = isNear ? '#00f3ff' : '#ff6600';

      // HUD Box Corners
      ctx.strokeStyle = accentColor;
      ctx.lineWidth = 2;
      ctx.shadowColor = accentColor;
      ctx.shadowBlur = 8;

      const cornerLen = Math.min(width, height) * 0.25;

      // Top-Left
      ctx.beginPath();
      ctx.moveTo(x, y + cornerLen);
      ctx.lineTo(x, y);
      ctx.lineTo(x + cornerLen, y);
      ctx.stroke();

      // Top-Right
      ctx.beginPath();
      ctx.moveTo(x + width - cornerLen, y);
      ctx.lineTo(x + width, y);
      ctx.lineTo(x + width, y + cornerLen);
      ctx.stroke();

      // Bottom-Left
      ctx.beginPath();
      ctx.moveTo(x, y + height - cornerLen);
      ctx.lineTo(x, y + height);
      ctx.lineTo(x + cornerLen, y + height);
      ctx.stroke();

      // Bottom-Right
      ctx.beginPath();
      ctx.moveTo(x + width - cornerLen, y + height);
      ctx.lineTo(x + width, y + height);
      ctx.lineTo(x + width, y + height - cornerLen);
      ctx.stroke();

      // Label Tag
      ctx.fillStyle = 'rgba(10, 15, 28, 0.85)';
      ctx.fillRect(x, y - 24, 150, 20);
      ctx.fillStyle = accentColor;
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.fillText(`${p.label} (${Math.round(p.confidence * 100)}%)`, x + 6, y - 9);
    });
    ctx.restore();
  };

  const drawTrajectoryTail = (
    ctx: CanvasRenderingContext2D,
    history: Array<{ x: number; y: number; speed: number }>
  ) => {
    if (history.length < 2) return;
    ctx.save();

    for (let i = 1; i < history.length; i++) {
      const p1 = history[i - 1];
      const p2 = history[i];

      const progress = i / history.length; // 0..1
      const alpha = progress * 0.9;
      const lineWidth = 1 + progress * 5;

      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);

      // Color gradient based on ball speed
      const isHighSpeed = p2.speed > 80;
      const strokeStyle = isHighSpeed
        ? `rgba(255, 70, 0, ${alpha})`
        : `rgba(0, 243, 255, ${alpha})`;

      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = lineWidth;
      ctx.shadowColor = isHighSpeed ? '#ff4600' : '#00f3ff';
      ctx.shadowBlur = 12 * progress;
      ctx.lineCap = 'round';
      ctx.stroke();
    }

    ctx.restore();
  };

  const drawBallAndVector = (
    ctx: CanvasRenderingContext2D,
    ball: BallTelemetry,
    w: number,
    h: number,
    drawVector: boolean
  ) => {
    ctx.save();
    const bx = ball.x * w;
    const by = ball.y * h;

    // Ball Core (Glowing 40mm Ping Pong Ball)
    ctx.beginPath();
    ctx.arc(bx, by, 7, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.shadowColor = '#00f3ff';
    ctx.shadowBlur = 20;
    ctx.fill();

    // Outer Glow Ring
    ctx.beginPath();
    ctx.arc(bx, by, 12, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(0, 243, 255, 0.6)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Velocity Vector Arrow
    if (drawVector && (ball.vx !== 0 || ball.vy !== 0)) {
      const vectorScale = 3.0;
      const targetX = bx + ball.vx * vectorScale;
      const targetY = by + ball.vy * vectorScale;

      ctx.beginPath();
      ctx.moveTo(bx, by);
      ctx.lineTo(targetX, targetY);
      ctx.strokeStyle = '#ffcc00';
      ctx.lineWidth = 2;
      ctx.shadowColor = '#ffcc00';
      ctx.shadowBlur = 8;
      ctx.stroke();

      // Velocity Tag
      ctx.fillStyle = '#ffcc00';
      ctx.font = 'bold 11px "JetBrains Mono", monospace';
      ctx.fillText(`${ball.speed_kmh} km/h`, targetX + 6, targetY + 4);
    }

    ctx.restore();
  };

  const drawBounceRipples = (ctx: CanvasRenderingContext2D) => {
    ctx.save();
    bouncesRef.current.forEach((b, idx) => {
      ctx.beginPath();
      ctx.arc(b.x, b.y, b.radius, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255, 230, 0, ${b.alpha})`;
      ctx.lineWidth = 2.5;
      ctx.shadowColor = '#ffe600';
      ctx.shadowBlur = 15;
      ctx.stroke();

      // Update ripple expansion
      b.radius += 1.2;
      b.alpha -= 0.035;
    });

    // Clean up finished ripples
    bouncesRef.current = bouncesRef.current.filter((b) => b.alpha > 0);
    ctx.restore();
  };

  const drawHeatmap = (ctx: CanvasRenderingContext2D, points: Array<{ x: number; y: number }>) => {
    ctx.save();
    points.forEach((pt) => {
      const grad = ctx.createRadialGradient(pt.x, pt.y, 2, pt.x, pt.y, 18);
      grad.addColorStop(0, 'rgba(255, 60, 0, 0.7)');
      grad.addColorStop(0.5, 'rgba(255, 200, 0, 0.35)');
      grad.addColorStop(1, 'rgba(0, 0, 0, 0)');

      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 18, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();
    });
    ctx.restore();
  };

  return (
    <div className="relative w-full h-full min-h-[440px] bg-slate-950 rounded-2xl overflow-hidden border border-slate-800 shadow-2xl">
      {/* Background Graphic / Video Layer */}
      <div className="absolute inset-0 bg-gradient-to-b from-slate-900 via-slate-950 to-black opacity-90" />
      
      {/* Canvas Overlay Layer */}
      <canvas
        ref={canvasRef}
        className="relative z-10 w-full h-full block cursor-crosshair"
      />

      {/* Live Telemetry Overlay HUD */}
      {currentFrame && (
        <div className="absolute top-4 left-4 z-20 flex items-center gap-3 glass-panel px-4 py-2 rounded-xl text-xs font-mono border border-cyan-500/30">
          <div className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping" />
          <span className="text-slate-300">STREAM 120 FPS</span>
          <span className="text-slate-600">|</span>
          <span className="text-cyan-400 font-bold">
            {currentFrame.ball ? `${currentFrame.ball.speed_kmh} KM/H` : 'SEARCHING...'}
          </span>
          <span className="text-slate-600">|</span>
          <span className="text-amber-400 uppercase font-semibold">
            {currentFrame.spin.type} ({currentFrame.spin.rpm} RPM)
          </span>
        </div>
      )}
    </div>
  );
};
