'use strict';
const fs=require('node:fs');
const path=require('node:path');
const assert=require('node:assert/strict');
const {enginePacket,freePort}=require('../services.cjs');
const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function eventually(check,timeout=60000){const start=Date.now();let error;while(Date.now()-start<timeout){try{return await check();}catch(e){error=e;}await wait(200);}throw error||Error('Test timeout');}
async function captureSettled(window,verify){
  // A hidden window may return its previous compositor surface on the first
  // capture. Activate capture, then allow the asserted DOM/canvas to paint.
  await window.webContents.capturePage();
  await window.webContents.executeJavaScript('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))');
  await wait(250);
  await verify();
  return window.webContents.capturePage();
}
exports.run=async({app,window,engine,startSession,endSession,openRecording,setBallBackend,dialog,dataRoot,getOrigin,getContext})=>{
  const report={started_at:new Date().toISOString(),packaged:app.isPackaged,electron:process.versions.electron,checks:[],renderer_errors:[]};
  const output=process.env.TT_DESKTOP_SMOKE_REPORT;
    window.webContents.on('console-message',event=>{if(event.level==='error')report.renderer_errors.push(event.message);});
  window.webContents.on('render-process-gone',(_event,details)=>report.renderer_errors.push(details.reason));
  try{
    assert.ok(output&&process.env.TT_DESKTOP_SMOKE_VIDEO,'Explicit smoke fixture and report required');
    const security=await window.webContents.executeJavaScript(`({node:typeof require,bridge:typeof window.ttDesktop?.startLive,isolated:typeof process,launcher:!!document.getElementById('live')})`);
    assert.deepEqual(security,{node:'undefined',bridge:'function',isolated:'undefined',launcher:true});report.security=security;
    report.checks.push('Sandboxed native window with isolated, typed preload bridge');
    await window.webContents.executeJavaScript(`document.getElementById('review').click()`);
    await eventually(async()=>{assert.ok((await window.webContents.executeJavaScript('document.body.innerText')).includes('Training review'));assert.equal(engine.child,null);assert.equal(getContext().mode,null);assert.equal(getContext().phase,'idle');});
    report.checks.push('Launcher training review opens without starting an inference backend or camera');
    await window.loadFile(path.join(__dirname,'..','launcher.html'));
    await eventually(async()=>{assert.equal(await window.webContents.executeJavaScript(`!!document.getElementById('live')&&!document.getElementById('live').disabled`),true);});
    // Exercise the actual launcher button -> preload -> IPC -> C++ startup.
    await window.webContents.executeJavaScript(`document.getElementById('live').click()`);
    await eventually(async()=>{assert.equal(getContext().phase,'ready');assert.equal(getContext().mode,'live');});
    const first=await enginePacket(engine.url,null,engine.child.pid);
    assert.equal(first.mode,'live');assert.equal(first.camera_active,false);
    report.live={pid:engine.child.pid,url:engine.url,camera_active:first.camera_active};
    await eventually(async()=>{const html=await window.webContents.executeJavaScript('document.body.innerText');assert.ok(html.includes('AUTHORIZE WEBCAM'));assert.ok(html.includes('LOCAL APP'));});
    report.checks.push('Launcher live control starts only live mode; camera remains unauthorized');
    const livePid=engine.child.pid;
    await endSession();assert.equal(engine.child,null);
    assert.throws(()=>process.kill(livePid,0));report.checks.push('End session releases its backend process');
    // Automated native dialog plumbing: supply a Unicode/space-containing fixture
    // through the real openRecording handler; no camera or external files mutated.
    const originalDialog=dialog.showOpenDialog;
    dialog.showOpenDialog=async(_window,options)=>{assert.equal(options.properties[0],'openFile');assert.ok(options.filters[0].extensions.includes('mov'));return {canceled:false,filePaths:[process.env.TT_DESKTOP_SMOKE_VIDEO]};};
    try{
      await window.webContents.executeJavaScript(`document.getElementById('recording').click()`);
      await eventually(async()=>{assert.equal(getContext().phase,'ready');assert.equal(getContext().mode,'simulation');});
    }finally{dialog.showOpenDialog=originalDialog;}
    const simulationPid=engine.child.pid;
    let packet=await eventually(async()=>{const p=await enginePacket(engine.url,null,simulationPid);assert.equal(p.analysis?.status,'COMPLETE');assert.ok(p.frame_id>0);return p;},120000);
    assert.equal(packet.analysis.analyzed_frames,packet.analysis.total_frames);assert.equal(packet.mode,'simulation');
    report.recording={pid:simulationPid,url:engine.url,analysis:packet.analysis,source:packet.video_file||packet.source_name,source_id:packet.source_id};
    await enginePacket(engine.url,'PAUSE',simulationPid);await enginePacket(engine.url,'SEEK_FRAME:120',simulationPid);
    await eventually(async()=>{const text=await window.webContents.executeJavaScript('document.body.innerText');assert.ok(text.includes('FRAME 120'));});
    report.checks.push('Native recording selection bridge handles Unicode/spaces and analyzes every frame');
    const endpoints=['/','/training-review','/api/training-review','/favicon.ico'];
    report.http=[];
    for(const endpoint of endpoints){const response=await fetch(getOrigin()+endpoint);assert.equal(response.status,200);report.http.push({endpoint,status:response.status});}
    const snapshot=await window.webContents.executeJavaScript(`({toolbar:!!document.querySelector('[aria-label="Desktop session controls"]'),canvas:!!document.querySelector('canvas'),frame:document.body.innerText.match(/FRAME 120/)?120:null,buttons:[...document.querySelectorAll('button')].map(b=>b.textContent).filter(t=>/Open recording|Live camera|End session/.test(t))})`);
    assert.equal(snapshot.toolbar,true);assert.equal(snapshot.frame,120);report.rendered=snapshot;
    await window.webContents.executeJavaScript('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))');
    const painted=await window.webContents.executeJavaScript(`document.querySelector('canvas[aria-label="Source footage with synchronized ball, person and table detections"]').dataset.frameId`);
    assert.equal(painted,'120');report.rendered.painted_frame=Number(painted);
    fs.writeFileSync(output.replace(/\.json$/,'.png'),(await window.webContents.capturePage()).toPNG());
    report.checks.push('Desktop renders footage and native controls; review API, current assets and favicon load');
    await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(button=>button.textContent.includes('2D RADAR')).click()`);
    const radar=await eventually(async()=>{
      const value=await window.webContents.executeJavaScript(`(()=>{const element=document.querySelector('[aria-label="2D table plane projection"]');return element?{frame:element.dataset.frameId,x:element.dataset.ballX,y:element.dataset.ballY}:null})()`);
      assert.equal(value?.frame,'120');return value;
    });
    await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(button=>button.textContent.includes('3D VOLUMETRIC')).click()`);
    const volumetric=await eventually(async()=>{
      const value=await window.webContents.executeJavaScript(`(()=>{const element=document.querySelector('[aria-label="3D table plane projection"]');return element?{frame:element.dataset.frameId,x:element.dataset.ballX,y:element.dataset.ballY}:null})()`);
      assert.equal(value?.frame,'120');return value;
    });
    assert.deepEqual(volumetric,radar);report.spatial_views={radar,volumetric};
    report.checks.push('Footage, 2D and 3D retain the same paused frame and authoritative table coordinates');
    // Review data belongs only to this smoke test's explicit temporary root.
    // The renderer fetch supplies the genuine Chromium Origin header; the test
    // never injects Origin or edits a user's saved capture/review.
    const reviewClip='live_0_0',reviewDir=path.join(dataRoot,'training_data','live',reviewClip);
    fs.mkdirSync(reviewDir,{recursive:true});
    fs.writeFileSync(path.join(reviewDir,'capture.json'),JSON.stringify({source:'desktop_smoke_fixture'}));
    fs.writeFileSync(path.join(reviewDir,'frame_1.jpg'),(await window.webContents.capturePage()).toJPEG(85));
    fs.writeFileSync(path.join(reviewDir,'frames.jsonl'),JSON.stringify({file:'frame_1.jpg',source_frame:120,elapsed_ms:1000})+'\n');
    await window.loadURL(getOrigin()+'/training-review');
    await eventually(async()=>{assert.ok((await window.webContents.executeJavaScript('document.body.innerText')).includes('Training review'));});
    report.training_review=await window.webContents.executeJavaScript(`(async()=>{
      const read=await fetch('/api/training-review?clip=live_0_0');
      const detail=await read.json();
      const labels={ball:[[.2,.3,.01,.01]],person:[],table:[],geometry:{landmarks:{},landmarks_reviewed:false,visibility:'visible',blur:'unknown',events:[],events_reviewed:false}};
      const response=await fetch('/api/training-review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({clip:'live_0_0',file:'frame_1.jpg',scope:'ball',reviewed:true,revision:detail.revision,labels})});
      const result=await response.json();
      const persisted=await(await fetch('/api/training-review?clip=live_0_0')).json();
      return {origin:location.origin,status:response.status,saved:result.saved,error:result.error,annotation:persisted.annotations?.['frame_1.jpg'],revision:result.revision};
    })()`);
    assert.equal(report.training_review.origin,getOrigin());assert.equal(report.training_review.status,200,JSON.stringify(report.training_review));
    assert.equal(report.training_review.saved,true);assert.deepEqual(report.training_review.annotation.ball,[[.2,.3,.01,.01]]);
    assert.deepEqual(report.training_review.annotation.reviewed_classes,{ball:true,person:false,table:false});
    assert.equal(report.training_review.annotation.source_frame,120);
    assert.ok(fs.existsSync(path.join(reviewDir,'annotations.json')));
    report.checks.push('Actual Chromium renderer saves and reloads reviewed fixture annotations at its exact dynamic loopback origin');
    await window.loadURL(getOrigin()+'/training-review');
    await eventually(async()=>{const ready=await window.webContents.executeJavaScript(`(()=>{const select=document.querySelector('select[aria-label="Capture clip"]');if(!select?.querySelector('option[value="live_0_0"]'))return false;select.value='live_0_0';select.dispatchEvent(new Event('change',{bubbles:true}));return true;})()`);assert.equal(ready,true);});
    await eventually(async()=>{assert.equal(await window.webContents.executeJavaScript(`!!document.querySelector('img[alt="Clean source frame for annotation"]')?.naturalWidth`),true);});
    await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(b=>b.textContent==='Blur streak · drag endpoints').click();document.querySelector('[aria-label="Frame annotation area"]').scrollIntoView({block:'center'})`);
    await wait(100);
    const area=await window.webContents.executeJavaScript(`(()=>{const r=document.querySelector('[aria-label="Frame annotation area"]').getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height};})()`);
    const a={x:Math.round(area.x+area.width*.3),y:Math.round(area.y+area.height*.4)},b={x:Math.round(area.x+area.width*.4),y:Math.round(area.y+area.height*.45)};
    window.webContents.sendInputEvent({type:'mouseMove',...a});window.webContents.sendInputEvent({type:'mouseDown',button:'left',clickCount:1,...a});
    window.webContents.sendInputEvent({type:'mouseMove',...b});window.webContents.sendInputEvent({type:'mouseUp',button:'left',clickCount:1,...b});
    await eventually(async()=>{assert.ok((await window.webContents.executeJavaScript('document.body.innerText')).includes('Streak center'));});
    await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(b=>b.textContent==='Save this frame').click()`);
    report.streak_review=await eventually(async()=>{const detail=await(await fetch(getOrigin()+'/api/training-review?clip=live_0_0')).json();const annotation=detail.annotations?.['frame_1.jpg'];assert.equal(annotation?.geometry?.blur_streak?.reviewed,true);assert.deepEqual(annotation.ball,[[.2,.3,.01,.01]]);return annotation.geometry.blur_streak;});
    assert.ok(Math.abs(report.streak_review.center[0]-.35)<.003);assert.ok(Math.abs(report.streak_review.center[1]-.425)<.003);
    assert.ok(report.streak_review.half_length_px>0);assert.equal(report.streak_review.method,'manual_streak_endpoints');
    const verifyStreak=async()=>{const state=await window.webContents.executeJavaScript(`({clip:document.querySelector('select[aria-label="Capture clip"]')?.value,loaded:!!document.querySelector('img[alt="Clean source frame for annotation"]')?.naturalWidth,text:document.body.innerText})`);assert.equal(state.clip,'live_0_0');assert.equal(state.loaded,true);assert.ok(state.text.includes('saved reviewed label'));assert.ok(state.text.includes('Ball label saved.'));};
    await eventually(verifyStreak);
    fs.writeFileSync(output.replace(/\.json$/,'.streak.png'),(await captureSettled(window,verifyStreak)).toPNG());
    report.checks.push('Actual pointer drag creates a reviewed source-size streak through GUI Save, preserving prior boxes');
    // Bulk no-ball review is exercised only on an isolated synthetic import.
    // The user's new recording and annotations are never changed by this test.
    const absenceClip='phone_abcdef0123456789_1',absenceDir=path.join(dataRoot,'training_data','live',absenceClip);
    fs.mkdirSync(absenceDir,{recursive:true});
    fs.writeFileSync(path.join(absenceDir,'capture.json'),JSON.stringify({source:'recorded_camera',session_id:'SYNTHETIC_SMOKE_ONLY',original_sha256:'a'.repeat(64),full_source_decode_counted:true,source_frames:3,sampled_frames:3,source_fps_average:30}));
    const absenceJpeg=(await window.webContents.capturePage()).toJPEG(85);
    const absenceRows=[1,2,3].map(id=>({file:`frame_${id}.jpg`,source_frame:id,elapsed_ms:(id-1)*1000/30,burst_id:1}));
    const priorAbsence={};
    for(const row of absenceRows){fs.writeFileSync(path.join(absenceDir,row.file),absenceJpeg);priorAbsence[row.file]={reviewed:false,reviewed_classes:{ball:false,person:false,table:false},ball:[],person:[],table:[]};}
    fs.writeFileSync(path.join(absenceDir,'frames.jsonl'),absenceRows.map(row=>JSON.stringify(row)+'\n').join(''));
    fs.writeFileSync(path.join(absenceDir,'annotations.json'),JSON.stringify(priorAbsence));
    await window.loadURL(getOrigin()+'/training-review');
    await eventually(async()=>{assert.equal(await window.webContents.executeJavaScript(`(()=>{const s=document.querySelector('select[aria-label="Capture clip"]');if(!s?.querySelector('option[value="${absenceClip}"]'))return false;s.value='${absenceClip}';s.dispatchEvent(new Event('change',{bubbles:true}));return true;})()`),true);});
    await eventually(async()=>{assert.ok((await window.webContents.executeJavaScript('document.body.innerText')).includes('No-ball interval'));});
    await window.webContents.executeJavaScript(`[...document.querySelectorAll('details')].find(d=>d.querySelector('summary')?.textContent.includes('No-ball interval')).open=true;[...document.querySelectorAll('button')].find(b=>b.textContent==='Preview no-ball interval').click()`);
    await eventually(async()=>{assert.ok((await window.webContents.executeJavaScript('document.body.innerText')).includes('3 saved image(s) selected'));});
    const beforeAbsence=JSON.parse(fs.readFileSync(path.join(absenceDir,'annotations.json'),'utf8'));assert.deepEqual(beforeAbsence,priorAbsence);
    assert.equal(await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(b=>b.textContent==='Confirm ball absence for selected images').disabled`),true);
    await window.webContents.executeJavaScript(`document.querySelector('input[aria-label="Confirm no ball in selected images"]').click()`);
    await eventually(async()=>{assert.equal(await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(b=>b.textContent==='Confirm ball absence for selected images').disabled`),false);});
    await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(b=>b.textContent==='Confirm ball absence for selected images').click()`);
    const verifyAbsence=async()=>{const text=await window.webContents.executeJavaScript('document.body.innerText');assert.ok(text.includes('Ball absence saved for 3 sampled images'));};
    await eventually(verifyAbsence);
    const persistedAbsence=JSON.parse(fs.readFileSync(path.join(absenceDir,'annotations.json'),'utf8'));
    for(const row of absenceRows){const label=persistedAbsence[row.file];assert.equal(label.geometry.visibility,'absent');assert.deepEqual(label.ball,[]);assert.deepEqual(label.reviewed_classes,{ball:true,person:false,table:false});assert.equal(label.ball_absence_review.session_id,'SYNTHETIC_SMOKE_ONLY');assert.equal(label.ball_absence_review.start_source_frame,1);assert.equal(label.ball_absence_review.end_source_frame,3);assert.ok(label.ball_absence_review.image_sha256);}
    fs.writeFileSync(output.replace(/\.json$/,'.absence.png'),(await captureSettled(window,verifyAbsence)).toPNG());
    report.absence_review={sampled_frames:3,explicit_gui_confirmation:true,scene_classes_unreviewed:true,interval_review:persistedAbsence['frame_1.jpg'].ball_absence_review};
    report.checks.push('Packaged Chromium previews and explicitly confirms a source-bound no-ball interval without reviewing scene classes');
    await window.loadURL(getOrigin()+'/');
    const draft={ball:[[.2,.3,.01,.01]],person:[],table:[]},draftKey='phone_0123456789abcdef_1/frame_000001.jpg';
    await window.webContents.executeJavaScript(`window.ttDesktop.saveReviewDrafts(${JSON.stringify({[draftKey]:draft})})`);
    const saved=await window.webContents.executeJavaScript('window.ttDesktop.getReviewDrafts()');assert.deepEqual(saved[draftKey],draft);
    report.checks.push('Review drafts persist in an application-owned file independent of the HTTP port');
    // Invalid recordings preserve the previous source, just as in browser mode.
    const bad=path.join(dataRoot,'invalid recording.mov');fs.writeFileSync(bad,'not a video');
    const beforeInvalid=await enginePacket(engine.url,null,simulationPid);
    await enginePacket(engine.url,`LOAD_VIDEO:${bad}`,simulationPid);
    packet=await eventually(async()=>{const p=await enginePacket(engine.url,null,simulationPid);assert.ok(p.video_error);return p;});
    assert.equal(packet.source_id,beforeInvalid.source_id);assert.equal(packet.analysis_id,beforeInvalid.analysis_id);assert.equal(packet.frame_id,beforeInvalid.frame_id);report.checks.push('Invalid video preserves the active recording');
    const cancelledDialog=dialog.showOpenDialog;
    dialog.showOpenDialog=async()=>({canceled:true,filePaths:[]});
    try{await openRecording();assert.equal(engine.child.pid,simulationPid);}finally{dialog.showOpenDialog=cancelledDialog;}
    report.checks.push('Canceling recording selection preserves the session');
    if(process.env.TT_DESKTOP_BLURBALL_VIDEO){
      await startSession('simulation',process.env.TT_DESKTOP_BLURBALL_VIDEO);
      const original=await eventually(async()=>{const p=await enginePacket(engine.url,null,engine.child.pid);assert.equal(p.source_video,path.basename(process.env.TT_DESKTOP_BLURBALL_VIDEO));assert.equal(p.analysis.status,'COMPLETE');return p;},120000);
      const oldPid=engine.child.pid;
      await eventually(async()=>{const exists=await window.webContents.executeJavaScript(`!!document.querySelector('select[aria-label="Ball model"]')`);assert.equal(exists,true);});
      await window.webContents.executeJavaScript(`(()=>{const select=document.querySelector('select[aria-label="Ball model"]');select.value='blurball';select.dispatchEvent(new Event('change',{bubbles:true}));})()`);
      await eventually(async()=>{assert.equal(getContext().ballBackend,'blurball');assert.equal(getContext().phase,'ready');});
      const temporal=await eventually(async()=>{const p=await enginePacket(engine.url,null,engine.child.pid);assert.equal(p.analysis?.status,'COMPLETE');assert.equal(p.ball_model?.backend,'blurball');return p;},120000);
      assert.notEqual(engine.child.pid,oldPid);assert.throws(()=>process.kill(oldPid,0));
      assert.notEqual(temporal.analysis_id,original.analysis_id);assert.equal(temporal.analysis.analyzed_frames,temporal.total_frames);
      assert.equal(temporal.ball_model.lookahead_frames,2);assert.equal(temporal.ball_model.score_calibrated,false);
      await enginePacket(engine.url,'PAUSE',engine.child.pid);await enginePacket(engine.url,'SEEK_FRAME:120',engine.child.pid);
      const selected=await eventually(async()=>{const p=await enginePacket(engine.url,null,engine.child.pid);assert.equal(p.frame_id,120);assert.ok(p.ball_blur);assert.ok(p.ball);return p;});
      await eventually(async()=>{const painted=await window.webContents.executeJavaScript(`(()=>{const canvas=document.querySelector('canvas[aria-label="Source footage with synchronized ball, person and table detections"]');return {frame:canvas?.dataset.frameId,analysis:canvas?.dataset.analysisId,backend:canvas?.dataset.ballBackend,text:document.body.innerText}})()`);assert.equal(painted.frame,'120');assert.equal(painted.analysis,selected.analysis_id);assert.equal(painted.backend,'blurball');assert.ok(painted.text.includes('FRAME 120'));assert.ok(painted.text.includes('COMPLETE · 272'));});
      await window.webContents.executeJavaScript(`document.querySelector('canvas[aria-label="Source footage with synchronized ball, person and table detections"]').scrollIntoView({block:'center'})`);
      await window.webContents.executeJavaScript('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))');
      fs.writeFileSync(output.replace(/\.json$/,'.blurball.png'),(await window.webContents.capturePage()).toPNG());
      const timeline=fs.readFileSync(path.join(dataRoot,'artifacts','analyses',temporal.analysis_id,'frames.jsonl'),'utf8').trim().split('\n').map(line=>JSON.parse(line));
      const forecastFrame=timeline.find(row=>row.frame_id>=30&&row.ball_flight?.status==='EXPERIMENTAL_FORECAST');assert.ok(forecastFrame,'No supported experimental forecast in fixture');
      await enginePacket(engine.url,`SEEK_FRAME:${forecastFrame.frame_id}`,engine.child.pid);
      await eventually(async()=>{const p=await enginePacket(engine.url,null,engine.child.pid);assert.equal(p.frame_id,forecastFrame.frame_id);assert.equal(p.ball_flight?.status,'EXPERIMENTAL_FORECAST');});
      await eventually(async()=>{const checkbox=await window.webContents.executeJavaScript(`!!document.querySelector('input[aria-label="Experimental trajectory forecast"]')`);assert.equal(checkbox,true);});
      await window.webContents.executeJavaScript(`document.querySelector('input[aria-label="Experimental trajectory forecast"]').click()`);
      const paintedForecast=await eventually(async()=>{const value=await window.webContents.executeJavaScript(`(()=>{const c=document.querySelector('canvas[aria-label="Source footage with synchronized ball, person and table detections"]');return {frame:c?.dataset.frameId,analysis:c?.dataset.analysisId,points:JSON.parse(c?.dataset.forecastPoints||'[]')};})()`);assert.equal(value.frame,String(forecastFrame.frame_id));assert.equal(value.analysis,selected.analysis_id);assert.deepEqual(value.points,forecastFrame.ball_flight.points);return value;});
      assert.ok(forecastFrame.ball_flight.support_frames.every(id=>id<=forecastFrame.frame_id));
      report.forecast_render={...paintedForecast,provenance:forecastFrame.ball_flight};
      report.checks.push('Opt-in forecast renders only authoritative C++ points with matching analysis/frame and past observation support');
      const verifyForecast=async()=>{const state=await window.webContents.executeJavaScript(`({checked:document.querySelector('input[aria-label="Experimental trajectory forecast"]')?.checked,frame:document.querySelector('canvas[aria-label="Source footage with synchronized ball, person and table detections"]')?.dataset.frameId,text:document.body.innerText})`);assert.equal(state.checked,true);assert.equal(state.frame,String(forecastFrame.frame_id));assert.ok(state.text.includes('FRAME '+forecastFrame.frame_id));};
      fs.writeFileSync(output.replace(/\.json$/,'.forecast.png'),(await captureSettled(window,verifyForecast)).toPNG());
      await enginePacket(engine.url,'SEEK_FRAME:120',engine.child.pid);
      const spatial=[];
      for(const [button,label] of [['2D RADAR','2D table plane projection'],['3D VOLUMETRIC','3D table plane projection']]){
        await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(b=>b.textContent.includes(${JSON.stringify(button)})).click()`);
        spatial.push(await eventually(async()=>{const value=await window.webContents.executeJavaScript(`(()=>{const element=document.querySelector(${JSON.stringify('[aria-label="'+label+'"]')});return element?{frame:element.dataset.frameId,analysis:element.dataset.analysisId,x:element.dataset.ballX,y:element.dataset.ballY}:null})()`);assert.equal(value?.frame,'120');assert.equal(value?.analysis,selected.analysis_id);return value;}));
      }
      assert.deepEqual(spatial[0],spatial[1]);
      assert.equal(Number(spatial[0].x),selected.table_position[0]);assert.equal(Number(spatial[0].y),selected.table_position[1]);
      report.blurball={pid:engine.child.pid,url:engine.url,analysis:temporal.analysis,model:temporal.ball_model,frame:selected.frame_id,ball:selected.ball,blur:selected.ball_blur};
      report.blurball.spatial_views=spatial;
      report.checks.push('Rendered model selector restarts owned backend, isolates cache identity, covers all recording frames and paints BlurBall centroid/blur evidence');
      await setBallBackend('yolo');assert.equal(getContext().ballBackend,'yolo');
      assert.throws(()=>process.kill(report.blurball.pid,0));
      report.checks.push('Switching back to YOLO releases the experimental backend');
    }
    assert.equal(report.renderer_errors.length,0,JSON.stringify(report.renderer_errors));
    report.frontend_origin=getOrigin();report.data_root=dataRoot;report.ok=true;
    // The outer runner checks all children exited after the actual window closes.
    window.close();
  }catch(error){report.ok=false;report.error=error.stack;console.error(error);app.quit();}
  finally{report.finished_at=new Date().toISOString();if(output)fs.writeFileSync(output,JSON.stringify(report,null,2));}
};
