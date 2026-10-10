export const landmarkNames=['corner_near_left','corner_near_right','corner_far_right','corner_far_left','center_near','center_far','net_left','net_right'] as const;
export type LandmarkName=typeof landmarkNames[number];
export const eventNames=['serve','strike','table_bounce','net_contact','edge_bounce','rally_start','rally_end'] as const;
export type ReviewPoint=[number,number];
export type BlurStreak={
  endpoints:[ReviewPoint,ReviewPoint];
  source_width:number;
  source_height:number;
  center:ReviewPoint;
  half_length_px:number;
  /** Undirected source-image axis in [0, PI); unresolved for a point. */
  axis_rad:number|null;
  method:'manual_streak_endpoints';
  reviewed:boolean;
};
export type ReviewGeometry={
  landmarks:Partial<Record<LandmarkName,[number,number]>>;
  landmarks_reviewed:boolean;
  visibility:'unknown'|'visible'|'partial'|'occluded'|'absent';
  blur:'unknown'|'none'|'mild'|'moderate'|'severe'|'extreme';
  events:Array<typeof eventNames[number]>;
  events_reviewed:boolean;
  blur_streak?:BlurStreak|null;
};
export const emptyGeometry=():ReviewGeometry=>({landmarks:{},landmarks_reviewed:false,visibility:'unknown',blur:'unknown',events:[],events_reviewed:false});
function reviewPoint(value:unknown):ReviewPoint{
  if(!Array.isArray(value)||value.length!==2||value.some(x=>typeof x!=='number'||!Number.isFinite(x)||x<0||x>1))throw Error('Invalid normalized blur endpoint');
  return [value[0],value[1]];
}
function imageDimension(value:unknown):number{
  if(typeof value!=='number'||!Number.isInteger(value)||value<=0||value>32768)throw Error('Invalid blur source dimensions');
  return value;
}
export function createBlurStreak(from:ReviewPoint,to:ReviewPoint,size:{width:number;height:number},reviewed=false):BlurStreak{
  const a=reviewPoint(from),b=reviewPoint(to),width=imageDimension(size.width),height=imageDimension(size.height);
  if(typeof reviewed!=='boolean')throw Error('Explicit blur review state required');
  const dx=(b[0]-a[0])*width,dy=(b[1]-a[1])*height,length=Math.hypot(dx,dy);
  return {endpoints:[a,b],source_width:width,source_height:height,center:[(a[0]+b[0])/2,(a[1]+b[1])/2],half_length_px:length/2,
    axis_rad:length===0?null:((Math.atan2(dy,dx)%Math.PI)+Math.PI)%Math.PI,method:'manual_streak_endpoints',reviewed};
}
export function validateBlurStreak(value:unknown):BlurStreak|null{
  if(value===undefined||value===null)return null;
  if(!value||typeof value!=='object')throw Error('Invalid blur streak');
  const v=value as BlurStreak;
  if(v.method!=='manual_streak_endpoints'||!Array.isArray(v.endpoints)||v.endpoints.length!==2)throw Error('Invalid manual blur endpoints');
  // Derived fields are regenerated. Client-provided centers/lengths/angles are
  // never authoritative, including recovered browser drafts.
  return createBlurStreak(v.endpoints[0],v.endpoints[1],{width:v.source_width,height:v.source_height},v.reviewed);
}
/** Source JPEG dimensions, without trusting a rendered or capture-manifest size. */
export function jpegDimensions(bytes:Uint8Array):{width:number;height:number}{
  if(bytes.length<4||bytes[0]!==0xff||bytes[1]!==0xd8)throw Error('Blur review requires a valid source JPEG');
  const startOfFrame=new Set([0xc0,0xc1,0xc2,0xc3,0xc5,0xc6,0xc7,0xc9,0xca,0xcb,0xcd,0xce,0xcf]);
  let offset=2;
  while(offset<bytes.length){
    if(bytes[offset++]!==0xff)throw Error('Invalid source JPEG marker');
    while(offset<bytes.length&&bytes[offset]===0xff)offset++;
    const marker=bytes[offset++];
    if(marker===0xd9||marker===0xda)break;
    if(marker===0x01||(marker>=0xd0&&marker<=0xd7))continue;
    if(offset+2>bytes.length)break;
    const length=bytes[offset]*256+bytes[offset+1];
    if(length<2||offset+length>bytes.length)throw Error('Truncated source JPEG');
    if(startOfFrame.has(marker)){
      if(length<8)throw Error('Invalid source JPEG dimensions');
      return {height:imageDimension(bytes[offset+3]*256+bytes[offset+4]),width:imageDimension(bytes[offset+5]*256+bytes[offset+6])};
    }
    offset+=length;
  }
  throw Error('Source JPEG dimensions unavailable');
}
export function validateGeometry(value:unknown):ReviewGeometry{
  if(value===undefined)return emptyGeometry();
  if(!value||typeof value!=='object')throw Error('Invalid geometry review');
  const v=value as ReviewGeometry;
  if(!v.landmarks||Array.isArray(v.landmarks)||typeof v.landmarks!=='object'||typeof v.landmarks_reviewed!=='boolean'||typeof v.events_reviewed!=='boolean')throw Error('Explicit geometry review states required');
  const result=emptyGeometry();
  for(const [name,point] of Object.entries(v.landmarks)){
    if(!landmarkNames.includes(name as LandmarkName)||!Array.isArray(point)||point.length!==2||point.some(x=>typeof x!=='number'||!Number.isFinite(x)||x<0||x>1))throw Error('Invalid normalized table landmark');
    result.landmarks[name as LandmarkName]=[point[0],point[1]];
  }
  if(!['unknown','visible','partial','occluded','absent'].includes(v.visibility)||!['unknown','none','mild','moderate','severe','extreme'].includes(v.blur))throw Error('Invalid visibility/blur tag');
  if(!Array.isArray(v.events)||v.events.some(e=>!eventNames.includes(e))||new Set(v.events).size!==v.events.length)throw Error('Invalid event labels');
  return {...result,landmarks_reviewed:v.landmarks_reviewed,events_reviewed:v.events_reviewed,visibility:v.visibility,blur:v.blur,events:v.events,
    ...(v.blur_streak!==undefined?{blur_streak:validateBlurStreak(v.blur_streak)}:{})};
}
