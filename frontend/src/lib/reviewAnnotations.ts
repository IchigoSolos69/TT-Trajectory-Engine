import {ReviewGeometry,validateGeometry} from './reviewGeometry';
export type Kind='ball'|'person'|'table';
export type Box=[number,number,number,number];
export type Labels=Record<Kind,Box[]>&{geometry?:ReviewGeometry};
export type Annotation=Labels&{reviewed:boolean;reviewed_classes?:Record<Kind,boolean>};
export type Frame={file:string;source_frame:number;burst_id?:number;burst_index?:number;burst_length?:number;review_metadata?:{reasons?:string[];suggestions?:Labels;ball_model_sha256?:string}};
export const emptyLabels=():Labels=>({ball:[],person:[],table:[]});
export const classReviewed=(value:Annotation|undefined,kind:Kind)=>value?.reviewed_classes?.[kind]??value?.reviewed===true;
export function ballReviewProblem(labels:Labels):string{
  const visibility=labels.geometry?.visibility??'unknown';
  if(labels.ball.length&&visibility==='absent')return 'Ball boxes conflict with “Not in frame”.';
  if(labels.ball.length&&visibility==='occluded')return 'A fully hidden ball cannot have an observed box. Correct visibility or remove the box.';
  if(labels.geometry?.blur_streak&&['absent','occluded','unknown'].includes(visibility))return 'A blur label needs visible or partly visible ball evidence. Clear the streak or correct visibility.';
  if(!labels.ball.length&&!labels.geometry?.blur_streak&&['visible','partial'].includes(visibility))return 'The ball is tagged visible but has no box. Place its box, label its streak, or correct the visibility.';
  if(!labels.ball.length&&visibility==='unknown')return 'Click the ball, mark “Not in frame”, or skip if you cannot tell.';
  return '';
}
export function previousBallSuggestion(frames:Frame[],index:number,annotations:Record<string,Annotation>):Box|null{
  const current=frames[index],previous=frames[index-1];
  if(!current||!previous||current.burst_id===undefined||current.burst_id!==previous.burst_id||current.source_frame!==previous.source_frame+1)return null;
  const label=annotations[previous.file];
  if(!classReviewed(label,'ball')||!label||label.ball.length!==1||ballReviewProblem(label)||label.geometry?.visibility==='occluded')return null;
  return [...label.ball[0]];
}
export function nextBallReviewIndex(frames:Frame[],index:number,annotations:Record<string,Annotation>):number{
  for(let step=1;step<=frames.length;step++){
    const candidate=(index+step)%frames.length,label=annotations[frames[candidate].file];
    if(!classReviewed(label,'ball')||ballReviewProblem(label??emptyLabels()))return candidate;
  }
  return -1;
}
export function validDraft(value:unknown):value is Labels{
  if(!value||typeof value!=='object')return false;
  const label=value as Labels;
  try{validateGeometry(label.geometry);}catch{return false;}
  return (['ball','person','table'] as Kind[]).every(kind=>Array.isArray(label[kind])&&label[kind].every(box=>Array.isArray(box)&&box.length===4&&box.every(n=>typeof n==='number'&&Number.isFinite(n))&&box[0]>=0&&box[1]>=0&&box[2]>0&&box[3]>0&&box[0]+box[2]<=1.000001&&box[1]+box[3]<=1.000001));
}
