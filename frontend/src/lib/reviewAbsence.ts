import {Annotation, classReviewed} from './reviewAnnotations';
import {validateGeometry} from './reviewGeometry';

export type IndexedReviewFrame={file:string;source_frame:number;elapsed_ms:number};
export type AbsenceInterval={start_source_frame:number;end_source_frame:number};
export type AbsenceProvenance=AbsenceInterval&{
  method:'manual_interval';review_id:string;reviewed_at:string;
  original_sha256:string;session_id:string;index_sha256:string;
  previous_annotation_revision:string;sampled_frames:number;
};

export function previewAbsenceInterval(frames:IndexedReviewFrame[],annotations:Record<string,Annotation>,interval:AbsenceInterval){
  const {start_source_frame:start,end_source_frame:end}=interval;
  if(!Number.isSafeInteger(start)||!Number.isSafeInteger(end)||start<1||end<start)throw Error('Choose a valid increasing source-frame interval.');
  const names=new Set<string>();let previousFrame=0,previousTime=-1;
  for(const frame of frames){
    if(!/^frame_[0-9]+\.jpg$/.test(frame.file)||names.has(frame.file)||!Number.isSafeInteger(frame.source_frame)||frame.source_frame<=previousFrame||!Number.isFinite(frame.elapsed_ms)||frame.elapsed_ms<0||frame.elapsed_ms<=previousTime)throw Error('Capture index must contain unique chronological frames and timestamps.');
    names.add(frame.file);previousFrame=frame.source_frame;previousTime=frame.elapsed_ms;
  }
  const selected=frames.filter(frame=>frame.source_frame>=start&&frame.source_frame<=end);
  if(!selected.length)throw Error('No saved images fall inside this interval.');
  if(selected.length>5000)throw Error('Review at most 5,000 saved images in one interval.');
  const conflicts=selected.filter(frame=>{
    const annotation=annotations[frame.file];
    return annotation&&(annotation.ball?.length||annotation.geometry?.blur_streak||['visible','partial','occluded'].includes(annotation.geometry?.visibility??'unknown'));
  }).map(frame=>({file:frame.file,source_frame:frame.source_frame}));
  return {selected,conflicts};
}

export function reviewedAbsence(previous:Annotation|undefined,frame:IndexedReviewFrame,provenance:AbsenceProvenance,imageSha256:string){
  const geometry=validateGeometry(previous?.geometry);
  return {
    ...previous,
    reviewed:true,
    reviewed_classes:{ball:true,person:classReviewed(previous,'person'),table:classReviewed(previous,'table')},
    ball:[],person:previous?.person??[],table:previous?.table??[],
    geometry:{...geometry,visibility:'absent' as const,blur:'unknown' as const,blur_streak:null},
    source_frame:frame.source_frame,elapsed_ms:frame.elapsed_ms,reviewed_at:provenance.reviewed_at,
    ball_absence_review:{...provenance,image_sha256:imageSha256},
  };
}
