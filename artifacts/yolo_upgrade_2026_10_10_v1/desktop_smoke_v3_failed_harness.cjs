'use strict';
const fs=require('node:fs');
const path=require('node:path');
const assert=require('node:assert/strict');
const {createHash}=require('node:crypto');
const vm=require('node:vm');
const {execFileSync}=require('node:child_process');
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
    // Paired-session and temporal-continuity fixtures remain under the explicit
    // smoke root. These are synthetic UI examples, never training evidence or
    // ball-accuracy measurements. Distinct compositor captures avoid passing a
    // temporal audit using three copies of one JPEG.
    const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
    const pairedRoot=path.join(dataRoot,'training_data','live');
    const pairedFixtures=[],pairedSession='SMOKE_PAIRED_A';
    for(const [device,fixtureClip]of [['iphone','phone_1111222233334444_1'],['pc_camera','live_400_0']]){
      const directory=path.join(pairedRoot,fixtureClip);fs.mkdirSync(directory,{recursive:true});
      const rows=[1,2,3].map(id=>({file:`frame_${id}.jpg`,source_frame:id,elapsed_ms:(id-1)*1000/60,burst_id:1,burst_index:id-1,burst_length:3}));
      const savedImages=[],labels={};let imageSize;
      for(const row of rows){
        await window.webContents.executeJavaScript(`(()=>{let mark=document.getElementById('paired-smoke-capture-marker');if(!mark){mark=document.createElement('div');mark.id='paired-smoke-capture-marker';document.body.appendChild(mark);}Object.assign(mark.style,{position:'fixed',top:'20px',left:${JSON.stringify((40+row.source_frame*90)+'px')},zIndex:'99999',background:${JSON.stringify(device==='iphone'?'#0891b2':'#7c3aed')},color:'white',padding:'20px',borderRadius:'12px'});mark.textContent=${JSON.stringify('SYNTHETIC '+device+' FRAME '+row.source_frame)};})()`);
        const verifyMarker=async()=>assert.equal(await window.webContents.executeJavaScript(`document.getElementById('paired-smoke-capture-marker')?.textContent`),'SYNTHETIC '+device+' FRAME '+row.source_frame);
        const image=await captureSettled(window,verifyMarker),bytes=image.toJPEG(90);imageSize=image.getSize();
        fs.writeFileSync(path.join(directory,row.file),bytes);savedImages.push(digest(bytes));
        labels[row.file]={reviewed:true,reviewed_classes:{ball:true,person:false,table:false},ball:[[.2,.3,.01,.01]],person:[],table:[],source_frame:row.source_frame,elapsed_ms:row.elapsed_ms,geometry:{landmarks:{},landmarks_reviewed:false,visibility:'visible',blur:'none',events:[],events_reviewed:false,...(row.source_frame===3?{blur_streak:{endpoints:[[.205,.305],[.205,.305]],source_width:imageSize.width,source_height:imageSize.height,center:[.205,.305],half_length_px:0,axis_rad:null,method:'manual_streak_endpoints',reviewed:true}}:{})}};
      }
      assert.equal(new Set(savedImages).size,3,'Three temporal fixture JPEGs must be distinct compositor captures');
      await window.webContents.executeJavaScript(`document.getElementById('paired-smoke-capture-marker')?.remove()`);
      const manifest=device==='iphone'?{source:'recorded_camera',session_id:pairedSession,original_path:'Synthetic smoke iPhone.MOV',original_sha256:'b'.repeat(64),full_source_decode_counted:true,source_frames:3,sampled_frames:3,source_fps_average:60,width:imageSize.width,height:imageSize.height}:{source:'live',camera_index:0,width:imageSize.width,height:imageSize.height,burst_frames:3};
      fs.writeFileSync(path.join(directory,'capture.json'),JSON.stringify(manifest));
      fs.writeFileSync(path.join(directory,'frames.jsonl'),rows.map(row=>JSON.stringify(row)+'\n').join(''));
      fs.writeFileSync(path.join(directory,'annotations.json'),JSON.stringify(labels));
      pairedFixtures.push({device,clip:fixtureClip,directory,rows,source_hashes:Object.fromEntries(['capture.json','frames.jsonl','annotations.json',...rows.map(row=>row.file)].map(file=>[file,digest(fs.readFileSync(path.join(directory,file)))]))});
    }
    const chooseReviewClip=async fixtureClip=>{
      await window.loadURL(getOrigin()+'/training-review');
      await eventually(async()=>{assert.equal(await window.webContents.executeJavaScript(`(()=>{const select=document.querySelector('select[aria-label="Capture clip"]');if(!select?.querySelector('option[value="${fixtureClip}"]'))return false;select.value='${fixtureClip}';select.dispatchEvent(new Event('change',{bubbles:true}));return true;})()`),true);});
      await eventually(async()=>{const state=await window.webContents.executeJavaScript(`({clip:document.querySelector('select[aria-label="Capture clip"]')?.value,loaded:!!document.querySelector('img[alt="Clean source frame for annotation"]')?.naturalWidth,session:!!document.querySelector('input[aria-label="Whole gameplay session ID"]'),disabled:document.querySelector('select[aria-label="Session role"]')?.disabled})`);assert.equal(state.clip,fixtureClip);assert.equal(state.loaded,true);assert.equal(state.session,true);assert.equal(state.disabled,false);});
    };
    const registryPath=path.join(pairedRoot,'training_sessions.json');
    const assignThroughGui=async({clip:fixtureClip,sessionId,stage,device})=>{
      await chooseReviewClip(fixtureClip);
      await window.webContents.executeJavaScript(`(()=>{const input=document.querySelector('input[aria-label="Whole gameplay session ID"]');if(!input.disabled){Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(input,${JSON.stringify(sessionId)});input.dispatchEvent(new Event('input',{bubbles:true}));input.dispatchEvent(new Event('change',{bubbles:true}));}const role=document.querySelector('select[aria-label="Session role"]');role.value=${JSON.stringify(stage)};role.dispatchEvent(new Event('change',{bubbles:true}));const device=document.querySelector('select[aria-label="Recording device"]');device.value=${JSON.stringify(device)};device.dispatchEvent(new Event('change',{bubbles:true}));})()`);
      await eventually(async()=>{const state=await window.webContents.executeJavaScript(`({session:document.querySelector('input[aria-label="Whole gameplay session ID"]')?.value,stage:document.querySelector('select[aria-label="Session role"]')?.value,device:document.querySelector('select[aria-label="Recording device"]')?.value,confirmDisabled:document.querySelector('input[aria-label="Confirm whole-session role and device"]')?.disabled})`);assert.equal(state.session,sessionId);assert.equal(state.stage,stage);assert.equal(state.device,device);assert.equal(state.confirmDisabled,false);});
      assert.equal(await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(button=>button.textContent==='Save session & device assignment').disabled`),true);
      await window.webContents.executeJavaScript(`document.querySelector('input[aria-label="Confirm whole-session role and device"]').click()`);
      await eventually(async()=>assert.equal(await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(button=>button.textContent==='Save session & device assignment').disabled`),false));
      await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(button=>button.textContent==='Save session & device assignment').click()`);
      await eventually(async()=>{const registry=JSON.parse(fs.readFileSync(registryPath,'utf8'));assert.equal(registry.captures[fixtureClip].session_id,sessionId);assert.equal(registry.captures[fixtureClip].domain,device);assert.equal(registry.sessions[sessionId].stage,stage);assert.equal(registry.sessions[sessionId].role,stage==='C'?'test':stage==='B'?'val':'train');assert.equal(registry.sessions[sessionId].sealed,stage==='C');});
      await eventually(async()=>{const state=await window.webContents.executeJavaScript(`({stage:document.querySelector('select[aria-label="Session role"]')?.value,disabled:document.querySelector('select[aria-label="Session role"]')?.disabled,text:document.body.innerText})`);assert.equal(state.stage,stage);assert.equal(state.disabled,true);assert.ok(state.text.includes('Saved:'));});
    };
    const continuityChecks=[];
    for(const fixture of pairedFixtures){
      await assignThroughGui({clip:fixture.clip,sessionId:pairedSession,stage:'A',device:fixture.device});
      await eventually(async()=>assert.equal(await window.webContents.executeJavaScript(`document.querySelectorAll('[aria-label="Consecutive burst review"] img').length`),3));
      await eventually(async()=>assert.equal(await window.webContents.executeJavaScript(`document.querySelector('input[aria-label="Confirm viewed completed burst continuity"]')?.disabled`),false));
      assert.equal(await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(button=>button.textContent==='Save viewed burst continuity').disabled`),true);
      await window.webContents.executeJavaScript(`document.querySelector('input[aria-label="Confirm viewed completed burst continuity"]').click()`);
      await eventually(async()=>assert.equal(await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(button=>button.textContent==='Save viewed burst continuity').disabled`),false));
      await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(button=>button.textContent==='Save viewed burst continuity').click()`);
      const audit=await eventually(async()=>{const value=JSON.parse(fs.readFileSync(path.join(fixture.directory,'continuity_audit.json'),'utf8'));assert.equal(value.reviewed,true);assert.equal(value.index_sha256,digest(fs.readFileSync(path.join(fixture.directory,'frames.jsonl'))));assert.deepEqual(value.source_frame_ranges,[[1,3]]);return value;});
      assert.equal(audit.source_id,fixture.device==='iphone'?'b'.repeat(64):'capture:'+fixture.clip);
      for(const row of fixture.rows)assert.equal(audit.image_sha256[row.file],fixture.source_hashes[row.file]);
      await eventually(async()=>assert.ok((await window.webContents.executeJavaScript('document.body.innerText')).includes('Confirmed source frames 1–3')));
      // Reload verifies actual persisted metadata, rather than only local state.
      await window.loadURL(getOrigin()+'/training-review');
      await eventually(async()=>{assert.equal(await window.webContents.executeJavaScript(`(()=>{const select=document.querySelector('select[aria-label="Capture clip"]');if(!select?.querySelector('option[value="${fixture.clip}"]'))return false;select.value='${fixture.clip}';select.dispatchEvent(new Event('change',{bubbles:true}));return true;})()`),true);});
      await eventually(async()=>assert.ok((await window.webContents.executeJavaScript('document.body.innerText')).includes('Confirmed source frames 1–3')));
      continuityChecks.push({device:fixture.device,clip:fixture.clip,index_sha256:audit.index_sha256,source_id:audit.source_id,source_frame_ranges:audit.source_frame_ranges,reloaded:true});
    }
    const pairedReadiness=await eventually(async()=>{const value=await window.webContents.executeJavaScript(`(async()=>{const response=await fetch('/api/training-sessions');return {origin:location.origin,status:response.status,...await response.json()};})()`);assert.equal(value.origin,getOrigin());assert.equal(value.status,200);for(const device of ['iphone','pc_camera']){assert.equal(value.readiness.counts.train[device].visible_boxes,3);assert.equal(value.readiness.counts.train[device].positive_windows,1);}assert.equal(value.readiness.first_cycle_ready,false);assert.equal(value.readiness.counts.test,null);return value;});
    await eventually(async()=>{const text=await window.webContents.executeJavaScript(`document.querySelector('[aria-label="Paired training sessions and readiness"]')?.innerText`);assert.ok(text.includes('6 / 2,000'));assert.ok(text.includes('2 / 500'));assert.ok(text.includes('Unknown / 120 s'));assert.ok(text.includes('Awaiting audited training data'));});
    report.paired_training_review={fixture_only:true,origin:pairedReadiness.origin,registry_path:registryPath,session:pairedReadiness.registry.sessions[pairedSession],readiness:pairedReadiness.readiness,continuity_checks:continuityChecks};
    report.checks.push('Actual packaged GUI explicitly assigns paired A/iPhone and A/PC recordings, persists immutable roles/devices and renders source-audited collection progress');
    report.checks.push('Actual packaged GUI confirms distinct original burst images, persists index/source/image-bound continuity and reloads its audit without changing ball annotations');
    // A separate synthetic C clip exercises the blind annotation path. Readiness
    // must not inspect its labels/images or expose zero-valued test metrics.
    const sealedClip='phone_5555666677778888_1',sealedDirectory=path.join(pairedRoot,sealedClip),sealedSession='SMOKE_PAIRED_C';fs.mkdirSync(sealedDirectory,{recursive:true});
    const sealedRows=pairedFixtures[0].rows.map(row=>({...row,review_metadata:{reasons:['SYNTHETIC_SMOKE_ONLY'],suggestions:{ball:[[.1,.2,.01,.01]],person:[],table:[]}}}));
    fs.writeFileSync(path.join(sealedDirectory,'capture.json'),JSON.stringify({source:'recorded_camera',session_id:sealedSession,original_path:'Synthetic sealed test.MOV',original_sha256:'c'.repeat(64),full_source_decode_counted:true,source_frames:3,sampled_frames:3,source_fps_average:60}));
    fs.writeFileSync(path.join(sealedDirectory,'frames.jsonl'),sealedRows.map(row=>JSON.stringify(row)+'\n').join(''));
    fs.writeFileSync(path.join(sealedDirectory,'annotations.json'),'{}');
    for(const row of sealedRows)fs.copyFileSync(path.join(pairedFixtures[0].directory,row.file),path.join(sealedDirectory,row.file));
    const sealedHashes=Object.fromEntries(['capture.json','frames.jsonl','annotations.json',...sealedRows.map(row=>row.file)].map(file=>[file,digest(fs.readFileSync(path.join(sealedDirectory,file)))]));
    await assignThroughGui({clip:sealedClip,sessionId:sealedSession,stage:'C',device:'iphone'});
    const verifySealed=async()=>{const state=await window.webContents.executeJavaScript(`({clip:document.querySelector('select[aria-label="Capture clip"]')?.value,text:document.body.innerText,suggestionsDisabled:[...document.querySelectorAll('button')].find(button=>button.textContent==='Load model suggestions')?.disabled})`);assert.equal(state.clip,sealedClip);assert.ok(state.text.includes('Sealed test: manual annotation only.'));assert.ok(state.text.includes('Sealed · not audited before final acceptance'));assert.equal(state.suggestionsDisabled,true);};
    await eventually(verifySealed);
    report.sealed_test_review=await window.webContents.executeJavaScript(`(async()=>{const readiness=await(await fetch('/api/training-sessions')).json(),manual=await(await fetch('/api/training-review?clip=${sealedClip}')).json();return {origin:location.origin,seal:manual.sealed_test,test_counts:readiness.readiness.counts.test,test_status:readiness.readiness.sealed_test_readiness,suggestions:manual.frames.map(frame=>frame.review_metadata?.suggestions??null)};})()`);
    assert.equal(report.sealed_test_review.origin,getOrigin());assert.equal(report.sealed_test_review.seal,true);assert.equal(report.sealed_test_review.test_counts,null);assert.equal(report.sealed_test_review.test_status,'NOT_AUDITED');assert.deepEqual(report.sealed_test_review.suggestions,[null,null,null]);
    for(const fixture of pairedFixtures)for(const [file,hash]of Object.entries(fixture.source_hashes))assert.equal(digest(fs.readFileSync(path.join(fixture.directory,file))),hash,'Session/continuity GUI must preserve original fixture data and labels');
    for(const [file,hash]of Object.entries(sealedHashes))assert.equal(digest(fs.readFileSync(path.join(sealedDirectory,file))),hash,'Sealing must preserve original source and unreviewed labels');
    report.paired_training_review.source_and_labels_preserved=true;
    fs.writeFileSync(output.replace(/\.json$/,'.paired-review.png'),(await captureSettled(window,verifySealed)).toPNG());
    report.checks.push('Packaged GUI seals C with manual-only annotation, suppresses model suggestions and leaves automated test readiness unavailable');
    if(process.env.TT_DESKTOP_ABSENCE_SMOKE_VIDEO){
      // This is a separate genuinely ball-free synthetic original. Never use
      // the official positive ball clip as absence, even in an isolated test.
      const intervalVideo=path.resolve(process.env.TT_DESKTOP_ABSENCE_SMOKE_VIDEO),intervalClip='phone_777788889999aaaa_1',intervalDirectory=path.join(pairedRoot,intervalClip);
      assert.ok(path.isAbsolute(process.env.TT_TRAINING_PYTHON||''),'Explicit training Python required for absence fixture metadata');
      fs.mkdirSync(intervalDirectory,{recursive:true});
      const metadata=JSON.parse(execFileSync(process.env.TT_TRAINING_PYTHON,['-B','-c',String.raw`import cv2,json,sys
c=cv2.VideoCapture(sys.argv[1],cv2.CAP_FFMPEG);assert c.isOpened()
count=0;size=None
while True:
 ok,frame=c.read()
 if not ok:break
 count+=1
 if count==1:
  size=[frame.shape[1],frame.shape[0]];assert cv2.imwrite(sys.argv[2],frame,[cv2.IMWRITE_JPEG_QUALITY,95])
c.release();print(json.dumps({'source_frames':count,'source_size':size}))`,intervalVideo,path.join(intervalDirectory,'frame_1.jpg')],{windowsHide:true,encoding:'utf8'}));
      assert.equal(metadata.source_frames,90,'Full absence fixture must be the isolated ninety-frame ball-free original');
      const intervalSourceHash=digest(fs.readFileSync(intervalVideo));
      const intervalCapture={source:'recorded_camera',session_id:pairedSession,original_path:intervalVideo,original_sha256:intervalSourceHash,full_source_decode_counted:true,source_frames:metadata.source_frames,sampled_frames:1,source_fps_average:30,capture_note:'SYNTHETIC_BALL_FREE_SMOKE_ONLY'};
      const intervalIndex=JSON.stringify({file:'frame_1.jpg',source_frame:1,elapsed_ms:0})+'\n';
      const intervalLabels={'frame_1.jpg':{reviewed:false,reviewed_classes:{ball:false,person:false,table:false},ball:[],person:[[.1,.2,.2,.3]],table:[],geometry:{visibility:'unknown',blur:'unknown',blur_streak:null}}};
      fs.writeFileSync(path.join(intervalDirectory,'capture.json'),JSON.stringify(intervalCapture));fs.writeFileSync(path.join(intervalDirectory,'frames.jsonl'),intervalIndex);fs.writeFileSync(path.join(intervalDirectory,'annotations.json'),JSON.stringify(intervalLabels));
      const intervalBefore=Object.fromEntries(['capture.json','frames.jsonl','annotations.json','frame_1.jpg'].map(file=>[file,digest(fs.readFileSync(path.join(intervalDirectory,file)))]));
      await assignThroughGui({clip:intervalClip,sessionId:pairedSession,stage:'A',device:'iphone'});
      await window.webContents.executeJavaScript(`document.querySelector('[aria-label="Full original absence interval"]').open=true;[...document.querySelectorAll('button')].find(button=>button.textContent==='Check local training decoder').click()`);
      await eventually(async()=>assert.ok((await window.webContents.executeJavaScript('document.body.innerText')).includes('Sequential original-video decoder is available.')));
      await window.webContents.executeJavaScript(`(()=>{const input=document.querySelector('input[aria-label="Full interval last frame"]');Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(input,'60');input.dispatchEvent(new Event('input',{bubbles:true}));input.dispatchEvent(new Event('change',{bubbles:true}));})()`);
      assert.equal(await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(button=>button.textContent==='Prepare full original interval').disabled`),true);
      await window.webContents.executeJavaScript(`document.querySelector('input[aria-label="Confirm exact interval preparation"]').click()`);
      await eventually(async()=>assert.equal(await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(button=>button.textContent==='Prepare full original interval').disabled`),false));
      await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(button=>button.textContent==='Prepare full original interval').click()`);
      await eventually(async()=>assert.ok((await window.webContents.executeJavaScript('document.body.innerText')).includes('60 interval frames · 90 source frames decoded · 2.00 s')),120000);
      let fullJobs=await window.webContents.executeJavaScript(`(async()=>{const response=await fetch('/api/training-absence?clip=${intervalClip}');return {origin:location.origin,status:response.status,...await response.json()};})()`);
      assert.equal(fullJobs.origin,getOrigin());assert.equal(fullJobs.status,200);assert.equal(fullJobs.intervals.length,1);assert.equal(fullJobs.intervals[0].reviewed,false,'Preparation cannot assert absence');
      const fullId=fullJobs.intervals[0].interval_id,fullDirectory=path.join(pairedRoot,'absence_intervals',fullId);
      assert.equal(fs.existsSync(path.join(fullDirectory,'review.json')),false);
      await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(button=>button.textContent==='Inspect complete prepared interval').click()`);
      await eventually(async()=>assert.equal(await window.webContents.executeJavaScript(`!!document.querySelector('img[alt="Full original interval source frame 1"]')?.naturalWidth`),true));
      await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(button=>button.textContent==='Play every prepared frame').click()`);
      await eventually(async()=>assert.ok((await window.webContents.executeJavaScript(`document.querySelector('[aria-label="Complete absence interval inspection"]')?.innerText`)).includes('source frame 3')));
      await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(button=>button.textContent==='Pause inspection')?.click()`);
      assert.equal(await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(button=>button.textContent==='Save complete original absence review').disabled`),true);
      await window.webContents.executeJavaScript(`document.querySelector('input[aria-label="Watched entire original interval"]').click();document.querySelector('input[aria-label="Confirm original interval has no ball"]').click()`);
      assert.equal(await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(button=>button.textContent==='Save complete original absence review').disabled`),true);
      await window.webContents.executeJavaScript(`document.querySelector('input[aria-label="Confirm no hidden or unsure ball"]').click()`);
      await eventually(async()=>assert.equal(await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(button=>button.textContent==='Save complete original absence review').disabled`),false));
      await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(button=>button.textContent==='Save complete original absence review').click()`);
      await eventually(async()=>assert.ok((await window.webContents.executeJavaScript('document.body.innerText')).includes('Full original absence review saved: 2.00 seconds.')));
      const fullReview=JSON.parse(fs.readFileSync(path.join(fullDirectory,'review.json'),'utf8')),fullManifest=JSON.parse(fs.readFileSync(path.join(fullDirectory,'manifest.json'),'utf8'));
      assert.equal(fullReview.source_sha256,intervalSourceHash);assert.equal(fullReview.manifest_sha256,digest(fs.readFileSync(path.join(fullDirectory,'manifest.json'))));assert.equal(fullReview.index_sha256,digest(fs.readFileSync(path.join(fullDirectory,'frames.jsonl'))));assert.equal(fullReview.source_index_sha256,digest(fs.readFileSync(path.join(fullDirectory,'source_frames.jsonl'))));assert.equal(fullManifest.source_decoded_frames,90);assert.equal(fullManifest.interval_frames,60);assert.equal(fullManifest.coverage_complete,true);
      for(const flag of ['confirm_entire_interval','confirm_no_ball','no_hidden_or_unsure','watched_full_original_interval'])assert.equal(fullReview[flag],true);
      // Delay an old inspection response, switch capture through the actual
      // renderer, and ensure the old interval cannot appear in the new source.
      await window.webContents.executeJavaScript(`(()=>{window.__fullIntervalOriginalFetch=window.fetch;window.fetch=async(...args)=>{const result=await window.__fullIntervalOriginalFetch(...args);if(String(args[0]).includes('inspect=true'))await new Promise(resolve=>setTimeout(resolve,800));return result;};[...document.querySelectorAll('button')].find(button=>button.textContent==='Inspect complete prepared interval').click();const select=document.querySelector('select[aria-label="Capture clip"]');select.value='live_400_0';select.dispatchEvent(new Event('change',{bubbles:true}));})()`);
      await eventually(async()=>assert.equal(await window.webContents.executeJavaScript(`document.querySelector('select[aria-label="Capture clip"]')?.value`),'live_400_0'));
      await wait(1200);
      assert.equal(await window.webContents.executeJavaScript(`!!document.querySelector('[aria-label="Complete absence interval inspection"]')`),false);
      await window.webContents.executeJavaScript(`window.fetch=window.__fullIntervalOriginalFetch;delete window.__fullIntervalOriginalFetch`);
      await chooseReviewClip(intervalClip);
      await window.webContents.executeJavaScript(`document.querySelector('[aria-label="Full original absence interval"]').open=true`);
      await eventually(async()=>assert.ok((await window.webContents.executeJavaScript('document.body.innerText')).includes('EXPLICITLY REVIEWED ABSENCE')));
      await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(button=>button.textContent==='Inspect complete prepared interval').click()`);
      await eventually(async()=>assert.ok((await window.webContents.executeJavaScript('document.body.innerText')).includes('Immutable complete-original-interval absence review saved.')));
      const fullReadiness=await window.webContents.executeJavaScript(`(async()=>{const response=await fetch('/api/training-sessions');return {origin:location.origin,status:response.status,...await response.json()};})()`);
      assert.equal(fullReadiness.origin,getOrigin());assert.equal(fullReadiness.status,200);assert.ok(Math.abs(fullReadiness.readiness.counts.train.iphone.verified_absence_seconds-2)<1e-7);assert.equal(fullReadiness.readiness.counts.test,null);
      for(const [file,hash]of Object.entries(intervalBefore))assert.equal(digest(fs.readFileSync(path.join(intervalDirectory,file))),hash,'Full original review must preserve old unknown scene labels/source bytes');
      report.full_original_absence={fixture_only:true,original:intervalVideo,source_sha256:intervalSourceHash,source_size:metadata.source_size,source_frames:90,interval_frames:60,duration_s:fullReview.duration_s,origin:getOrigin(),review:fullReview,source_and_scene_labels_preserved:true,reloaded:true,stale_inspection_rejected:true};
      fs.writeFileSync(output.replace(/\.json$/,'.full-absence.png'),(await captureSettled(window,async()=>assert.ok((await window.webContents.executeJavaScript('document.body.innerText')).includes('Immutable complete-original-interval absence review saved.')))).toPNG());
      report.checks.push('Packaged GUI prepares every original absence frame, accounts complete source PTS, requires all explicit assertions, persists review and credits exactly verified duration without changing prior scene labels');
      report.checks.push('Switching capture during delayed full-interval inspection rejects stale frame/gallery state and reloads immutable source-bound evidence');
    }
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

// Compile the evaluated renderer strings, not only this outer Node script.
// This standalone preflight opens no Electron window, service or camera.
// TypeScript is loaded only by the workspace CLI; packaged app.run has no
// dependency on the compiler or repository node_modules.
exports.compileRendererScripts=()=>{
  const ts=require('../../frontend/node_modules/typescript');
  const source=fs.readFileSync(__filename,'utf8'),tree=ts.createSourceFile(__filename,source,ts.ScriptTarget.Latest,true,ts.ScriptKind.JS);
  const expressions=[];
  const visit=node=>{
    if(ts.isCallExpression(node)&&ts.isPropertyAccessExpression(node.expression)&&node.expression.name.text==='executeJavaScript'){
      assert.equal(node.arguments.length,1,'Preflight expects one renderer-source argument');
      expressions.push({argument:node.arguments[0].getText(tree),line:tree.getLineAndCharacterOfPosition(node.getStart(tree)).line+1});
    }
    ts.forEachChild(node,visit);
  };
  visit(tree);
  assert.ok(expressions.length>60,'All smoke renderer expressions must be discovered');
  let compiled=0;
  for(const [device,stage]of [['iphone','A'],['pc_camera','A'],['iphone','B'],['iphone','C']]){
    const bindings={JSON,row:{source_frame:3},device,stage,fixtureClip:'phone_1111222233334444_1',sessionId:'SMOKE_PAIRED_'+stage,fixture:{clip:'live_400_0'},absenceClip:'phone_abcdef0123456789_1',sealedClip:'phone_5555666677778888_1',intervalClip:'phone_777788889999aaaa_1',draftKey:'phone_0123456789abcdef_1/frame_000001.jpg',draft:{ball:[[.2,.3,.01,.01]],person:[],table:[]},button:'2D RADAR',label:'2D table plane projection'};
    for(const expression of expressions){
      const script=new vm.Script('('+expression.argument+')',{filename:'smoke-argument-line-'+expression.line}).runInNewContext(bindings);
      assert.equal(typeof script,'string');
      new vm.Script(script,{filename:'smoke-renderer-line-'+expression.line});compiled++;
    }
  }
  return {passed:true,renderer_expressions:expressions.length,representative_variants:4,compiled,real_camera_opened:false,electron_started:false};
};
if(require.main===module&&process.argv.includes('--compile-renderer-scripts'))console.log(JSON.stringify(exports.compileRendererScripts()));
