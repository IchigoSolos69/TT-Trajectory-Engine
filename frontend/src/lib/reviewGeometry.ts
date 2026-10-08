export const landmarkNames=['corner_near_left','corner_near_right','corner_far_right','corner_far_left','center_near','center_far','net_left','net_right'] as const;
export type LandmarkName=typeof landmarkNames[number];
export const eventNames=['serve','strike','table_bounce','net_contact','edge_bounce','rally_start','rally_end'] as const;
export type ReviewGeometry={
  landmarks:Partial<Record<LandmarkName,[number,number]>>;
  landmarks_reviewed:boolean;
  visibility:'unknown'|'visible'|'partial'|'occluded'|'absent';
  blur:'unknown'|'none'|'mild'|'moderate'|'severe'|'extreme';
  events:Array<typeof eventNames[number]>;
  events_reviewed:boolean;
};
export const emptyGeometry=():ReviewGeometry=>({landmarks:{},landmarks_reviewed:false,visibility:'unknown',blur:'unknown',events:[],events_reviewed:false});
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
  return {...result,landmarks_reviewed:v.landmarks_reviewed,events_reviewed:v.events_reviewed,visibility:v.visibility,blur:v.blur,events:v.events};
}
