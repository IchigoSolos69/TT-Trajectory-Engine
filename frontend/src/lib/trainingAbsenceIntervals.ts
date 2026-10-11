import type {TrainingDomain,TrainingRole} from './trainingSessions';

export const validIntervalId=(value:unknown):value is string=>typeof value==='string'&&/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(value);
export const validSha=(value:unknown):value is string=>typeof value==='string'&&/^[0-9a-f]{64}$/.test(value);
export type FullAbsenceManifest={schema_version:1;status:'prepared';interval_id:string;clip:string;source_sha256:string;source_capture_sha256:string;original_path:string;resolved_source_path:string;session_id:string;domain:TrainingDomain;role:TrainingRole;stage:string;start_source_frame:number;end_source_frame:number;source_start_s:number;source_end_s:number;source_size:[number,number];interval_frames:number;source_decoded_frames:number;source_reported_frames:number;duration_s:number;coverage_complete:true;contains_ground_truth:false;timestamp_basis:string;index_sha256:string;source_index_sha256:string;created_at?:string;settings?:Record<string,unknown>};
export type FullAbsenceFrame={file:string;source_frame:number;source_timestamp_s:number;frame_duration_s:number;image_sha256:string;decoded_index:number;width:number;height:number};
export type FullAbsenceReview={schema_version:1;method:'manual_full_original_interval_absence';reviewed:true;confirm_entire_interval:true;confirm_no_ball:true;no_hidden_or_unsure:true;watched_full_original_interval:true;reviewed_at:string;interval_id:string;clip:string;source_capture_sha256:string;manifest_sha256:string;index_sha256:string;source_index_sha256:string;source_sha256:string;session_id:string;role:TrainingRole;domain:TrainingDomain;start_source_frame:number;end_source_frame:number;duration_s:number};
export type VerifiedAbsenceInterval={interval_id:string;clip:string;source_sha256:string;session_id:string;domain:TrainingDomain;role:'train'|'val';start_source_frame:number;end_source_frame:number;duration_s:number;frames:FullAbsenceFrame[]};

export function validateAbsenceRange(start:unknown,end:unknown){
  if(!Number.isSafeInteger(start)||!Number.isSafeInteger(end)||(start as number)<1||(end as number)<(start as number)||(end as number)-(start as number)+1>36000)throw Error('Choose an increasing original-source interval of at most 36,000 frames.');
  return {start_source_frame:start as number,end_source_frame:end as number};
}

/** Decode coverage is checked separately from the human assertion of absence. */
export function validateFullAbsence(manifest:FullAbsenceManifest,frames:FullAbsenceFrame[],sourceFrames:Array<{source_frame:number;source_timestamp_s:number;frame_duration_s:number}>,manifestSha:string,indexSha:string,sourceIndexSha:string,review?:FullAbsenceReview|null){
  if(manifest.schema_version!==1||manifest.status!=='prepared'||manifest.coverage_complete!==true||manifest.contains_ground_truth!==false||!validIntervalId(manifest.interval_id)||!validSha(manifest.source_sha256)||!validSha(manifest.source_capture_sha256)||manifest.index_sha256!==indexSha||manifest.source_index_sha256!==sourceIndexSha||!manifest.timestamp_basis||!Number.isFinite(manifest.duration_s)||manifest.duration_s<=0)throw Error('The interval needs a complete source-bound decoder audit.');
  const range=validateAbsenceRange(manifest.start_source_frame,manifest.end_source_frame);
  if(manifest.interval_frames!==range.end_source_frame-range.start_source_frame+1||frames.length!==manifest.interval_frames||sourceFrames.length!==manifest.source_decoded_frames||manifest.source_decoded_frames!==manifest.source_reported_frames||range.end_source_frame>manifest.source_decoded_frames)throw Error('Decoded source and selected interval frame counts do not reconcile.');
  let duration=0;
  for(let i=0;i<sourceFrames.length;i++){
    const frame=sourceFrames[i];
    if(frame.source_frame!==i+1||!Number.isFinite(frame.source_timestamp_s)||frame.source_timestamp_s<0||!Number.isFinite(frame.frame_duration_s)||frame.frame_duration_s<0||(i<sourceFrames.length-1&&(frame.frame_duration_s<=0||frame.frame_duration_s>.1||Math.abs(frame.frame_duration_s-(sourceFrames[i+1].source_timestamp_s-frame.source_timestamp_s))>1e-6))||(i===sourceFrames.length-1&&frame.frame_duration_s!==0)||(i>0&&frame.source_timestamp_s<=sourceFrames[i-1].source_timestamp_s))throw Error('Source timestamps must account for every sequential decoded frame.');
  }
  for(let i=0;i<frames.length;i++){
    const frame=frames[i],source=sourceFrames[range.start_source_frame+i-1];
    if(frame.file!==`frame_${frame.source_frame}.jpg`||frame.source_frame!==range.start_source_frame+i||frame.decoded_index!==frame.source_frame||!validSha(frame.image_sha256)||!Number.isSafeInteger(frame.width)||!Number.isSafeInteger(frame.height)||frame.width<1||frame.height<1||manifest.source_size?.[0]!==frame.width||manifest.source_size?.[1]!==frame.height||frame.source_timestamp_s!==source.source_timestamp_s||frame.frame_duration_s!==source.frame_duration_s)throw Error('Saved interval image lineage does not match the sequential source index.');
    duration+=frame.frame_duration_s;
  }
  if(Math.abs(duration-manifest.duration_s)>1e-6)throw Error('Reviewed interval duration disagrees with decoded timestamps.');
  const beginning=sourceFrames[range.start_source_frame-1].source_timestamp_s,ending=sourceFrames[Math.min(range.end_source_frame,sourceFrames.length-1)].source_timestamp_s;
  if(manifest.source_start_s!==beginning||manifest.source_end_s!==ending||Math.abs(ending-beginning-manifest.duration_s)>1e-6)throw Error('Original interval bounds disagree with decoded source timestamps.');
  if(review&&(typeof review.reviewed_at!=='string'||!Number.isFinite(Date.parse(review.reviewed_at))||!/(?:Z|[+-][0-9]{2}:[0-9]{2})$/.test(review.reviewed_at)))throw Error('Full interval review needs an explicit timestamp with timezone.');
  if(review&&(review.schema_version!==1||review.method!=='manual_full_original_interval_absence'||review.reviewed!==true||review.confirm_entire_interval!==true||review.confirm_no_ball!==true||review.no_hidden_or_unsure!==true||review.watched_full_original_interval!==true||!review.reviewed_at||review.interval_id!==manifest.interval_id||review.clip!==manifest.clip||review.source_capture_sha256!==manifest.source_capture_sha256||review.manifest_sha256!==manifestSha||review.index_sha256!==indexSha||review.source_index_sha256!==sourceIndexSha||review.source_sha256!==manifest.source_sha256||review.session_id!==manifest.session_id||review.role!==manifest.role||review.domain!==manifest.domain||review.start_source_frame!==range.start_source_frame||review.end_source_frame!==range.end_source_frame||review.duration_s!==manifest.duration_s))throw Error('The explicit absence review belongs to a changed source, interval or decoder audit.');
  return {duration_s:duration,frames:frames.length};
}

/** Merge overlapping source-frame intervals rather than counting a review twice. */
export function verifiedAbsenceSeconds(intervals:VerifiedAbsenceInterval[]){
  if(!intervals.length)return null;
  const covered=new Map<string,number>();
  for(const interval of intervals)for(const frame of interval.frames){
    const key=interval.source_sha256+':'+frame.source_frame,previous=covered.get(key);
    if(previous!==undefined&&Math.abs(previous-frame.frame_duration_s)>1e-9)throw Error('Overlapping absence intervals disagree on source timing.');
    covered.set(key,frame.frame_duration_s);
  }
  return Array.from(covered.values()).reduce((sum,duration)=>sum+duration,0);
}
