import {Annotation,classReviewed,Frame} from './reviewAnnotations';
import {VerifiedAbsenceInterval,verifiedAbsenceSeconds} from './trainingAbsenceIntervals';

export const trainingStages=['existing','A','B','C'] as const;
export const trainingDomains=['iphone','pc_camera'] as const;
export type TrainingStage=typeof trainingStages[number];
export type TrainingDomain=typeof trainingDomains[number];
export type TrainingRole='train'|'val'|'test';
export type TrainingSession={role:TrainingRole;stage:TrainingStage;sealed:boolean;assigned_at:string};
export type TrainingCaptureAssignment={session_id:string;domain:TrainingDomain;assigned_at:string;source_sha256?:string};
export type TrainingRegistry={schema_version:1;sessions:Record<string,TrainingSession>;captures:Record<string,TrainingCaptureAssignment>};
export const emptyTrainingRegistry=():TrainingRegistry=>({schema_version:1,sessions:{},captures:{}});
export const stageRole=(stage:TrainingStage):TrainingRole=>stage==='B'?'val':stage==='C'?'test':'train';
export const validCaptureName=(clip:string)=>/^(?:live_[0-9]+_[0-9]+|phone_[0-9a-f]{16}_[0-9]+)$/.test(clip);
const validSessionId=(value:unknown):value is string=>typeof value==='string'&&/^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$/.test(value)&&!['__proto__','constructor','prototype'].includes(value);

export function validateTrainingRegistry(value:unknown):TrainingRegistry{
  if(!value||typeof value!=='object')throw Error('Invalid training session registry');
  const registry=value as TrainingRegistry;
  if(registry.schema_version!==1||!registry.sessions||!registry.captures||Array.isArray(registry.sessions)||Array.isArray(registry.captures)||typeof registry.sessions!=='object'||typeof registry.captures!=='object')throw Error('Unsupported training session registry');
  for(const [id,session] of Object.entries(registry.sessions)){
    if(!validSessionId(id)||!session||!trainingStages.includes(session.stage)||session.role!==stageRole(session.stage)||session.sealed!==(session.role==='test')||typeof session.assigned_at!=='string')throw Error('Invalid training session role');
  }
  for(const [clip,capture] of Object.entries(registry.captures)){
    if(!validCaptureName(clip)||!capture||!validSessionId(capture.session_id)||!registry.sessions[capture.session_id]||!trainingDomains.includes(capture.domain)||typeof capture.assigned_at!=='string'||(capture.source_sha256!==undefined&&!/^[0-9a-f]{64}$/.test(capture.source_sha256)))throw Error('Invalid training capture assignment');
  }
  return registry;
}

/** Explicit assignment never alters the original capture manifest or labels. */
export function assignTrainingCapture(registry:TrainingRegistry,body:{clip:string;session_id:string;stage:TrainingStage;domain:TrainingDomain},originalSession:unknown,now:string,originalSource?:unknown):TrainingRegistry{
  validateTrainingRegistry(registry);
  if(!validCaptureName(body.clip)||!validSessionId(body.session_id)||!trainingStages.includes(body.stage)||!trainingDomains.includes(body.domain))throw Error('Choose an explicit session, stage and recording device');
  if(typeof originalSession==='string'&&originalSession.trim()&&originalSession!==body.session_id)throw Error('Keep the original recording session ID. Clips from one session cannot be reassigned to another.');
  if(originalSession==='iphone_same_venue_session_1'&&body.stage!=='existing')throw Error('The previously reviewed iPhone venue session is development/training data. It cannot become a new validation or sealed-test session.');
  const role=stageRole(body.stage),session=registry.sessions[body.session_id],previous=registry.captures[body.clip];
  if(session&&(session.role!==role||session.stage!==body.stage))throw Error('This session already has a fixed dataset role. Keep all its recordings in that role.');
  if(previous&&(previous.session_id!==body.session_id||previous.domain!==body.domain))throw Error('A saved capture assignment is fixed. Correct an assignment through an audited dataset migration, not by moving it between splits.');
  if(originalSource!==undefined&&!/^[0-9a-f]{64}$/.test(String(originalSource)))throw Error('Original recording source hash is invalid');
  const source=typeof originalSource==='string'?originalSource:undefined;
  if(previous?.source_sha256&&source!==previous.source_sha256)throw Error('The immutable original source identity changed');
  if(source)for(const existing of Object.values(registry.captures))if(existing.source_sha256===source&&(existing.session_id!==body.session_id||existing.domain!==body.domain||registry.sessions[existing.session_id].role!==role))throw Error('One original recording must keep one session, dataset role and recording device, including sealed test.');
  return {...registry,sessions:{...registry.sessions,[body.session_id]:session??{role,stage:body.stage,sealed:role==='test',assigned_at:now}},captures:{...registry.captures,[body.clip]:{...(previous??{session_id:body.session_id,domain:body.domain,assigned_at:now}),...(source?{source_sha256:source}:{})}}};
}
export function wholeSourceConflicts(registry:TrainingRegistry,captures:Array<{clip:string;original_sha256?:string}>){
  const bindings=new Map<string,Set<string>>(),sourceByClip=new Map<string,string>();
  for(const [clip,assignment]of Object.entries(registry.captures))if(assignment.source_sha256)sourceByClip.set(clip,assignment.source_sha256);
  for(const capture of captures){const assignment=registry.captures[capture.clip];if(!assignment||registry.sessions[assignment.session_id].sealed)continue;if(capture.original_sha256){if(assignment.source_sha256&&assignment.source_sha256!==capture.original_sha256)throw Error('Registry and original recording source identities disagree');sourceByClip.set(capture.clip,capture.original_sha256);}}
  for(const [clip,source]of Array.from(sourceByClip)){const assignment=registry.captures[clip],session=registry.sessions[assignment.session_id],identities=bindings.get(source)??new Set<string>();identities.add(JSON.stringify([assignment.session_id,session.role,assignment.domain]));bindings.set(source,identities);}
  return new Set(Array.from(bindings).filter(([,identities])=>identities.size>1).map(([source])=>source));
}

export type ReadinessFrame=Frame&{elapsed_ms?:number;image_sha256:string};
export type ContinuityAudit={schema_version:1;reviewed:true;index_sha256:string;source_id:string;source_frame_ranges:Array<[number,number]>;reviewed_at:string;method:'manual_saved_burst';image_sha256?:Record<string,string>};
export type ReadinessCapture={clip:string;original_sha256?:string;original_session_id?:string;frames:ReadinessFrame[];annotations:Record<string,Annotation>;continuity?:ContinuityAudit|null};
export type TrainingCounts={visible_frames:number;visible_boxes:number;absent_frames:number;precise_points:number;positive_windows:number;unknown_frames:number;conflicting_frames:number;rallies:number|null;verified_absence_seconds:number|null};
export type TrainingReadiness={schema_version:1;counts:{train:Record<TrainingDomain,TrainingCounts>;val:Record<TrainingDomain,TrainingCounts>;test:null};sealed_test_readiness:'NOT_AUDITED';unassigned:TrainingCounts;sessions:Record<string,TrainingSession>;assigned_captures:number;unassigned_captures:number;duplicate_images:number;conflicting_source_frames:number;conflicting_original_sources:number;training_targets:{visible_boxes:2000;absent_frames:1000;positive_windows:500};evaluation_targets:{visible_per_device:1000;rallies_per_device:10;absence_seconds_per_device:120};first_cycle_ready:false;limitations:string[]};
const emptyCounts=():TrainingCounts=>({visible_frames:0,visible_boxes:0,absent_frames:0,precise_points:0,positive_windows:0,unknown_frames:0,conflicting_frames:0,rallies:null,verified_absence_seconds:null});
function annotationState(annotation:Annotation|undefined){
  if(!annotation||!classReviewed(annotation,'ball'))return 'unknown';
  const visibility=annotation.geometry?.visibility??'unknown';
  if(annotation.ball?.length===1&&!['absent','occluded'].includes(visibility))return 'visible';
  if(annotation.ball?.length===0&&visibility==='absent'&&!annotation.geometry?.blur_streak)return 'absent';
  if(annotation.ball?.length===0&&['visible','partial'].includes(visibility)&&annotation.geometry?.blur_streak?.reviewed)return 'point';
  return 'unknown';
}
function annotationSignature(annotation:Annotation|undefined){
  const state=annotationState(annotation);
  return state==='unknown'?null:JSON.stringify({state,ball:annotation?.ball??[],point:annotation?.geometry?.blur_streak?.reviewed?annotation.geometry.blur_streak.center:null});
}

/** Collection progress only: frozen dataset and model release audits are separate. */
export function trainingReadiness(registry:TrainingRegistry,captures:ReadinessCapture[],absenceIntervals:VerifiedAbsenceInterval[]=[]):TrainingReadiness{
  validateTrainingRegistry(registry);
  const counts={train:{iphone:emptyCounts(),pc_camera:emptyCounts()},val:{iphone:emptyCounts(),pc_camera:emptyCounts()},test:null};
  const unassigned=emptyCounts(),sources=new Map<string,Set<string>>(),conflicts=new Set<string>(),images=new Set<string>(),acceptedSources=new Set<string>(),rallyCounts=new Map<TrainingCounts,Set<string>>();
  const sourceKey=(capture:ReadinessCapture,frame:ReadinessFrame)=>`${capture.original_sha256??capture.clip}:${frame.source_frame}`;
  const wholeConflicts=wholeSourceConflicts(registry,captures);
  const developmentCaptures=captures.filter(capture=>{const assignment=registry.captures[capture.clip];return !assignment||registry.sessions[assignment.session_id].role!=='test';});
  for(const capture of developmentCaptures)for(const frame of capture.frames){
    const signature=annotationSignature(capture.annotations[frame.file]);
    if(signature){const id=sourceKey(capture,frame),set=sources.get(id)??new Set<string>();set.add(signature);sources.set(id,set);if(set.size>1)conflicts.add(id);}
  }
  let assignedCaptures=captures.filter(capture=>!!registry.captures[capture.clip]).length,duplicateImages=0;
  for(const capture of developmentCaptures){
    const assignment=registry.captures[capture.clip],session=assignment?registry.sessions[assignment.session_id]:undefined;
    if(assignment&&capture.original_session_id&&capture.original_session_id!==assignment.session_id)throw Error('Registry and original capture session IDs disagree');
    const bucket=assignment&&session&&session.role!=='test'?counts[session.role][assignment.domain]:unassigned;
    if(capture.original_sha256&&wholeConflicts.has(capture.original_sha256)){bucket.conflicting_frames+=capture.frames.length;continue;}
    const accepted=new Set<string>();
    for(const frame of capture.frames){
      const id=sourceKey(capture,frame),annotation=capture.annotations[frame.file];
      if(conflicts.has(id)){bucket.conflicting_frames++;continue;}
      const state=annotationState(annotation);
      if(state==='unknown'){bucket.unknown_frames++;continue;}
      if(images.has(frame.image_sha256)||acceptedSources.has(id)){duplicateImages++;continue;}
      images.add(frame.image_sha256);acceptedSources.add(id);accepted.add(frame.file);
      if(state==='visible'||state==='point')bucket.visible_frames++;
      if(state==='visible')bucket.visible_boxes++;
      if(state==='absent')bucket.absent_frames++;
      if(annotation?.geometry?.blur_streak?.reviewed&&['visible','partial'].includes(annotation.geometry.visibility))bucket.precise_points++;
      if(annotation?.geometry?.rally_reviewed===true&&annotation.geometry.rally_id){const rallies=rallyCounts.get(bucket)??new Set<string>();rallies.add(`${capture.original_sha256??capture.clip}:${annotation.geometry.rally_id}`);rallyCounts.set(bucket,rallies);bucket.rallies=rallies.size;}
    }
    for(let i=2;i<capture.frames.length;i++){
      const window=capture.frames.slice(i-2,i+1),last=window[2],label=capture.annotations[last.file];
      if(!accepted.has(last.file)||!label?.geometry?.blur_streak?.reviewed||!['visible','partial'].includes(label.geometry.visibility))continue;
      if(assignment&&session?.stage!=='A')continue;
      if(!capture.continuity||!window.every(frame=>capture.continuity!.source_frame_ranges.some(([start,end])=>start<=frame.source_frame&&frame.source_frame<=end)))continue;
      if(!window.every(frame=>annotationState(capture.annotations[frame.file])==='visible'))continue;
      if(window.some((frame,j)=>frame.burst_id===undefined||frame.burst_id!==last.burst_id||(j>0&&(frame.source_frame!==window[j-1].source_frame+1||frame.burst_index!==((window[j-1].burst_index??-2)+1)||!Number.isFinite(frame.elapsed_ms)||!Number.isFinite(window[j-1].elapsed_ms)||frame.elapsed_ms!<=window[j-1].elapsed_ms!||frame.elapsed_ms!-window[j-1].elapsed_ms!>100))))continue;
      if(new Set(window.map(frame=>frame.image_sha256)).size!==3||window.some(frame=>conflicts.has(sourceKey(capture,frame))))continue;
      bucket.positive_windows++;
    }
  }
  const acceptedIntervals=absenceIntervals.filter(interval=>{
    const assignment=registry.captures[interval.clip],session=assignment?registry.sessions[assignment.session_id]:undefined;
    return !!assignment&&!!session&&session.role!=='test'&&session.role===interval.role&&assignment.domain===interval.domain&&assignment.session_id===interval.session_id&&!wholeConflicts.has(interval.source_sha256);
  });
  for(const interval of acceptedIntervals){
    const assignment=registry.captures[interval.clip],session=assignment?registry.sessions[assignment.session_id]:undefined;
    if(!assignment||!session||session.role==='test'||session.role!==interval.role||assignment.domain!==interval.domain||assignment.session_id!==interval.session_id)continue;
    const bucket=counts[interval.role][interval.domain];
    for(const frame of interval.frames){
      const id=interval.source_sha256+':'+frame.source_frame;
      if(conflicts.has(id)){bucket.conflicting_frames++;continue;}
      if(images.has(frame.image_sha256)||acceptedSources.has(id)){duplicateImages++;continue;}
      images.add(frame.image_sha256);acceptedSources.add(id);bucket.absent_frames++;
    }
  }
  for(const role of ['train','val'] as const)for(const domain of trainingDomains)counts[role][domain].verified_absence_seconds=verifiedAbsenceSeconds(acceptedIntervals.filter(interval=>interval.role===role&&interval.domain===domain));
  return {schema_version:1,counts,sealed_test_readiness:'NOT_AUDITED',unassigned,sessions:registry.sessions,assigned_captures:assignedCaptures,unassigned_captures:captures.length-assignedCaptures,duplicate_images:duplicateImages,conflicting_source_frames:conflicts.size,conflicting_original_sources:wholeConflicts.size,training_targets:{visible_boxes:2000,absent_frames:1000,positive_windows:500},evaluation_targets:{visible_per_device:1000,rallies_per_device:10,absence_seconds_per_device:120},first_cycle_ready:false,limitations:['Collection progress is not an audited dataset freeze or a model accuracy result.','Sealed session C labels, image data and frame index are not opened by collection readiness. Its test counts remain unavailable until final acceptance.','One immutable original source must retain one session, role and device. Disjoint-frame reimports cannot create independent validation data; all conflicting source copies are excluded. Legacy sealed assignments without saved source hashes still need a provenance audit.','Rallies require explicit complete-original-rally review and source-specific IDs. No-ball duration is credited only for explicit complete-original-interval review with all decoded frames and immutable source/image/timestamp audit; sparse saved images cannot establish it.','Exact duplicate images and conflicting source labels are excluded here. Near-duplicate cross-split leakage, independent session capture, second-pass annotation review and frozen manifests still need audit.','Positive temporal windows require three consecutive reviewed box-positive original images, a saved continuity audit and an explicitly reviewed center/streak target on the latest frame. Box centers alone do not count as precise temporal targets.']};
}
