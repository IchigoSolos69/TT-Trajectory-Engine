'use client';
import {ReactNode,useRef,useState} from 'react';

type Point=[number,number];
type Size={width:number;height:number};
type Zoom='fit'|1|2;
type Props={src:string;enabled:boolean;children:ReactNode;onAnnotation:(from:Point,to:Point,size:Size)=>void};

export function ReviewFrameImage({src,enabled,children,onAnnotation}:Props){
  const [zoom,setZoom]=useState<Zoom>('fit');
  const [loaded,setLoaded]=useState<(Size&{src:string})|null>(null);
  const [failed,setFailed]=useState('');
  const activeSrc=useRef(src);activeSrc.current=src;
  const start=useRef<{point:Point;src:string;zoom:Zoom}|null>(null);
  const ready=loaded?.src===src;
  const point=(event:React.PointerEvent):Point=>{
    // The rectangle belongs to the full image, not its scroll viewport. This
    // preserves normalized source coordinates at every zoom/scroll position.
    const rect=event.currentTarget.getBoundingClientRect();
    return [Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width)),Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height))];
  };
  return <div className="space-y-2">
    <div className="flex flex-wrap gap-2 items-center text-xs">
      <span className="text-slate-400">Image zoom</span>
      {(['fit',1,2] as Zoom[]).map(value=><button key={value} type="button" aria-pressed={zoom===value} disabled={!ready}
        onClick={()=>{start.current=null;setZoom(value);}}
        className={'rounded-lg border px-3 py-2 '+(zoom===value?'border-cyan-400 text-cyan-300':'border-slate-700')}>
        {value==='fit'?'Fit':`${value*100}%`}
      </button>)}
      <span className="text-slate-400">Use 100% or 200% for tiny balls. Scroll to move around the image; drag the ball’s visible extent.</span>
    </div>
    {failed===src&&<p role="alert" className="text-amber-300 text-sm">Frame image failed to load. Annotation is disabled; reload this capture.</p>}
    <div className="overflow-auto max-h-[640px] rounded-lg border border-slate-800 bg-black/30" aria-label="Scrollable annotation image">
      <div tabIndex={0} aria-label="Frame annotation area" className={'relative w-fit select-none touch-none outline-cyan-500 '+(zoom==='fit'?'max-w-full mx-auto':'')}
        onPointerDown={event=>{if(!ready||!enabled)return;event.currentTarget.focus();event.currentTarget.setPointerCapture(event.pointerId);start.current={point:point(event),src,zoom};}}
        onPointerUp={event=>{const initial=start.current;start.current=null;if(!ready||!enabled||!initial||initial.src!==src||initial.zoom!==zoom||!loaded)return;onAnnotation(initial.point,point(event),loaded);}}
        onPointerCancel={()=>{start.current=null;}}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img key={src} draggable={false} src={src} alt="Clean source frame for annotation"
          onLoad={event=>{if(activeSrc.current!==src)return;const target=event.currentTarget;setLoaded({src,width:target.naturalWidth,height:target.naturalHeight});setFailed('');}}
          onError={()=>{if(activeSrc.current!==src)return;setLoaded(null);setFailed(src);start.current=null;}}
          className={zoom==='fit'?'max-h-[600px] max-w-full block':'block max-w-none'}
          style={zoom!=='fit'&&ready?{width:loaded.width*zoom,height:loaded.height*zoom}:undefined}/>
        <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 100 100" preserveAspectRatio="none">{children}</svg>
      </div>
    </div>
  </div>;
}
