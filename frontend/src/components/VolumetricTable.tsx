 'use client';
import React, { useEffect, useState } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Line } from '@react-three/drei';
import { tableToScene } from '../lib/spatialFrame';
import type { TrajectorySample } from '../types/telemetry';

interface VolumetricProps { x: number; y: number; hasBall?: boolean; frameId?: number; analysisId?: string; trajectory?: TrajectorySample[]; }
function UncertaintyCorridor({samples}:{samples:TrajectorySample[]}) {
  return <group>{samples.map((s,i)=>{
    const p=s.table_position,prev=samples[i-1];if(!p||s.state==='LOST')return null;
    const position=tableToScene(p[0],p[1]),color=s.state==='REACQUIRED'?'#34d399':s.accepted?'#22d3ee':'#fbbf24';
    return <group key={`${s.segment_id}:${s.frame_id}`}>
      {s.ellipse&&i%3===0&&<group position={[position[0],.775,position[2]]} rotation={[0,-s.ellipse[2],0]}>
        <mesh rotation={[-Math.PI/2,0,0]} scale={[s.ellipse[0],s.ellipse[1],1]}><circleGeometry args={[1,24]}/><meshBasicMaterial color={color} transparent opacity={.065} depthWrite={false}/></mesh>
      </group>}
      {prev?.table_position&&prev.segment_id===s.segment_id&&<Line points={[tableToScene(...prev.table_position),position]} color={color} lineWidth={2} dashed={!s.accepted} dashSize={.025} gapSize={.015}/>}
    </group>;
  })}</group>;
}
function View({preset}:{preset:string}) {
  const {camera}=useThree();
  useEffect(()=>{
    if(preset==='Top') camera.position.set(0,5,.01);
    else if(preset==='Side') camera.position.set(4,1.5,0);
    else camera.position.set(3,2.8,4);
    camera.lookAt(0,.76,0);
  },[camera,preset]);
  return <OrbitControls key={preset} makeDefault target={[0,.76,0]} minDistance={1.5} maxDistance={9} maxPolarAngle={Math.PI/2-.03} />;
}
function Table() {
  return <group>
    <mesh receiveShadow position={[0,.74,0]}><boxGeometry args={[1.525,.04,2.74]} /><meshStandardMaterial color="#155e63" roughness={.75} /></mesh>
    {[-.65,.65].flatMap(x=>[-1.1,1.1].map(z=><group key={`${x}:${z}`}>
      <mesh castShadow position={[x,.35,z]}><boxGeometry args={[.055,.70,.055]} /><meshStandardMaterial color="#475569" metalness={.7} roughness={.4}/></mesh>
      <mesh position={[x,.045,z]} rotation={[Math.PI/2,0,0]}><cylinderGeometry args={[.045,.045,.055,16]}/><meshStandardMaterial color="#111827"/></mesh>
    </group>))}
    <Line points={[[-.7525,.765,-1.36],[.7525,.765,-1.36],[.7525,.765,1.36],[-.7525,.765,1.36],[-.7525,.765,-1.36]]} color="#f1f5f9" lineWidth={2}/>
    <Line points={[[0,.766,-1.36],[0,.766,1.36]]} color="#f1f5f9" lineWidth={1}/>
    {[-.86,.86].map(x=><mesh key={x} position={[x,.82,0]}><cylinderGeometry args={[.012,.012,.23,12]}/><meshStandardMaterial color="#cbd5e1"/></mesh>)}
    {Array.from({length:7},(_,i)=><Line key={`h${i}`} points={[[-.86,.76+i*.0254,0],[.86,.76+i*.0254,0]]} color={i===6?'#f8fafc':'#94a3b8'} lineWidth={i===6?3:.5}/>)}
    {Array.from({length:55},(_,i)=><Line key={`v${i}`} points={[[-.86+i*1.72/54,.76,0],[-.86+i*1.72/54,.9125,0]]} color="#94a3b8" lineWidth={.5}/>)}
    <mesh receiveShadow rotation={[-Math.PI/2,0,0]}><planeGeometry args={[12,12]}/><meshStandardMaterial color="#0f172a" roughness={1}/></mesh>
    <gridHelper args={[12,24,'#334155','#1e293b']} position={[0,.001,0]}/>
  </group>;
}
export default function VolumetricTable({x,y,hasBall=false,frameId=0,analysisId='',trajectory=[]}:VolumetricProps) {
  const [preset,setPreset]=useState('Perspective');
  const position = tableToScene(x,y);
  return <div aria-label="3D table plane projection" data-analysis-id={analysisId} data-frame-id={frameId} data-ball-x={hasBall ? x : ''} data-ball-y={hasBall ? y : ''} className="w-full h-[560px] rounded-2xl overflow-hidden border border-slate-800 relative bg-slate-950">
    <div className="absolute top-4 left-4 right-4 z-10 flex flex-wrap justify-between gap-2">
      <div className="bg-slate-950/85 rounded-lg p-3 text-xs font-mono text-cyan-300">3D TABLE VIEW · 2.74 × 1.525 m · NET 15.25 cm</div>
      <div className="flex gap-1">{['Perspective','Top','Side'].map(view=><button key={view} onClick={()=>setPreset(view)} className={`px-3 py-2 text-xs rounded-lg ${preset===view?'bg-cyan-400 text-slate-950':'bg-slate-900 text-slate-300'}`}>{view}</button>)}</div>
    </div>
    <Canvas shadows camera={{position:[3,2.8,4],fov:42}}>
      <color attach="background" args={['#020617']}/>
      <ambientLight intensity={1.1}/><directionalLight position={[2,6,3]} intensity={3} castShadow shadow-mapSize={[1024,1024]}/>
      <Table/>
      <UncertaintyCorridor samples={trajectory}/>
      {hasBall && <group>
        <mesh position={position} castShadow><sphereGeometry args={[.02,24,24]}/><meshStandardMaterial color="#fff7ed" roughness={.6}/></mesh>
        <mesh position={[position[0],.768,position[2]]} rotation={[-Math.PI/2,0,0]}><ringGeometry args={[.025,.04,32]}/><meshBasicMaterial color="#fbbf24" transparent opacity={.7}/></mesh>
      </group>}
      <View preset={preset}/>
    </Canvas>
    <div className="absolute bottom-4 left-4 right-4 rounded-lg bg-slate-950/85 p-3 font-mono text-xs text-slate-300 space-y-1">
      <p>{hasBall?`Frame ${frameId} · table projection: x ${x.toFixed(2)} m · y ${y.toFixed(2)} m`:'Waiting for a detected ball and table surface'}</p>
      <p className="text-slate-500">Same projection as 2D; height is unobserved. Marker lies on the table plane. Drag to orbit · scroll to zoom.</p>
    </div>
  </div>;
}
