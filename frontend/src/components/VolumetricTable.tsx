'use client';

import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment, Grid } from '@react-three/drei';
import * as THREE from 'three';

interface VolumetricProps {
  x: number; // 0.0 to 1.525 meters
  y: number; // 0.0 to 2.74 meters
  z?: number; // Optional height in meters (default ~0.15m above table)
}

function Ball({ x, y, z = 0.15 }: VolumetricProps) {
  const meshRef = useRef<THREE.Mesh>(null);

  // Smoothly interpolate 3D position at 120Hz frame loop
  useFrame(() => {
    if (meshRef.current) {
      // Map 2D (X, Y) meters to 3D (X, Z) axes. Center table origin at [0, 0, 0].
      const targetX = x - 1.525 / 2;
      const targetZ = y - 2.74 / 2;
      const targetY = Math.max(0.02, z); // Real-world parabolic height from C++ gravity engine

      meshRef.current.position.lerp(new THREE.Vector3(targetX, targetY, targetZ), 0.5);
    }
  });

  return (
    <mesh ref={meshRef} castShadow>
      {/* ITTF 40mm ball radius = 0.02m */}
      <sphereGeometry args={[0.02, 32, 32]} />
      <meshStandardMaterial color="#f97316" emissive="#ea580c" emissiveIntensity={0.6} roughness={0.1} />
    </mesh>
  );
}

export default function VolumetricTable({ x, y, z = 0.15 }: VolumetricProps) {
  return (
    <div className="w-full h-[520px] bg-slate-950 rounded-2xl overflow-hidden border border-slate-800 relative shadow-2xl">
      {/* 3D Viewport Controls Hint */}
      <div className="absolute top-4 left-4 z-10 glass-panel px-3 py-1.5 rounded-lg border border-slate-800 text-[10px] font-mono text-cyan-400 flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
        VOLUMETRIC 3D SCENE • ROTATE & PAN WITH MOUSE
      </div>

      <Canvas shadows camera={{ position: [0, 2.2, 3.8], fov: 45 }}>
        <Environment preset="city" />
        <ambientLight intensity={0.6} />
        <spotLight position={[0, 6, 0]} intensity={2.5} castShadow shadow-mapSize={[1024, 1024]} />

        {/* 3D ITTF Table Block */}
        <mesh receiveShadow position={[0, -0.05, 0]}>
          <boxGeometry args={[1.525, 0.1, 2.74]} />
          <meshStandardMaterial color="#0f172a" roughness={0.15} metalness={0.8} />
        </mesh>

        {/* ITTF Net Geometry */}
        <mesh position={[0, 0.076, 0]}>
          <boxGeometry args={[1.625, 0.1525, 0.005]} />
          <meshStandardMaterial color="#ffffff" opacity={0.7} transparent roughness={0.4} />
        </mesh>

        {/* Table Grid Surface Overlay */}
        <Grid
          position={[0, 0.001, 0]}
          args={[1.525, 2.74]}
          cellSize={0.5}
          cellColor="#06b6d4"
          sectionColor="#1e3a8a"
          fadeDistance={6}
        />

        {/* Live Telemetry Ball 3D Position */}
        <Ball x={x} y={y} z={z} />

        <OrbitControls makeDefault maxPolarAngle={Math.PI / 2 - 0.05} />
      </Canvas>
    </div>
  );
}
