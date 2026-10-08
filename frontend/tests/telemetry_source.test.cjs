const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const ts = require('../node_modules/typescript');
const states = [], effects = [], sockets = [], timers = new Map();
let timerId = 0;
const react = {
  useState(value) { const id = states.push(value) - 1; return [value, next => { states[id] = typeof next === 'function' ? next(states[id]) : next; }]; },
  useEffect(fn) { effects.push(fn); },
  useCallback(fn) { return fn; },
  useRef(value) { return { current: value }; },
};
class Socket {
  static OPEN = 1;
  readyState = 1;
  constructor() { sockets.push(this); }
  close() { this.readyState = 3; this.onclose?.(); }
  send() {}
}
const images=[], decodedFrames=new WeakMap();
class TestImage { constructor(){images.push(this);} }
const context = { exports: {}, require: name => name==='react'?react:{decodedFrames}, WebSocket: Socket, Image:TestImage,
  console: { error() {} }, fetch() { throw new Error('Mock footage must never be fetched'); },
  setTimeout(fn) { timers.set(++timerId, fn); return timerId; }, clearTimeout(id) { timers.delete(id); } };
const source = fs.readFileSync(require('node:path').join(__dirname, '../src/hooks/useTelemetry.ts'), 'utf8');
vm.runInNewContext(ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText, context);
context.exports.useTelemetry();
const cleanup = effects[0]();
const first = sockets[0];
first.onopen();
assert.equal(states[2], true);
first.onmessage({ data: JSON.stringify({ mode: 'live', frame_id: 1, ball: null, rally: { stroke_count: 3 } }) });
assert.equal(states[1], false);
assert.equal(states[3].rallyStrokes, 3);
const late = first.onmessage;
first.close();
assert.equal(states[0], null);
assert.equal(states[1], false);
assert.equal(states[2], false);
assert.equal(states[3].rallyStrokes, 0);
assert.equal(timers.size, 1);
late({ data: JSON.stringify({ mode: 'simulation', frame_id: 99 }) });
assert.equal(states[0], null);
const timer = [...timers.values()][0]; timers.clear(); timer();
const second = sockets[1];
second.onopen();
second.onmessage({ data: JSON.stringify({ mode: 'simulation', frame_id: 1, ball: null }) });
assert.equal(states[1], true);
assert.equal(states[0].frame_id, 1);
second.onmessage({data:JSON.stringify({mode:'simulation',frame_id:20,ball:null,impact_points:[[.2,.4],[.5,1.8]]})});
assert.equal(states[4].length,2,'cumulative impacts survive skipped previews even without a current ball');
second.onmessage({data:JSON.stringify({mode:'simulation',frame_id:20,ball:null,impact_points:[[.2,.4],[.5,1.8]]})});
assert.equal(states[4].length,2,'paused replay must not duplicate impacts');
second.onmessage({data:JSON.stringify({mode:'simulation',source_id:1,frame_id:21,preview_jpeg:'first'})});
assert.equal(states[0].frame_id,20,'telemetry must wait for matching image decode');
second.onmessage({data:JSON.stringify({mode:'simulation',source_id:1,frame_id:22,preview_jpeg:'second'})});
second.onmessage({data:JSON.stringify({mode:'simulation',source_id:1,frame_id:23,preview_jpeg:'latest'})});
images[0].onload();
assert.equal(states[0].frame_id,21);
assert.equal(images[1].src,'data:image/jpeg;base64,latest','decode queue keeps only newest pending image');
images[1].onload();
assert.equal(states[0].frame_id,23);
assert(decodedFrames.has(states[0]),'published frame has a decoded image for synchronous canvas rendering');
second.onmessage({data:JSON.stringify({mode:'simulation',source_id:1,frame_id:24,preview_jpeg:'stale'})});
second.onmessage({data:JSON.stringify({mode:'simulation',source_id:2,frame_id:1,preview_jpeg:'new recording'})});
images[2].onload();
assert.equal(states[0].frame_id,23,'old source decode must not publish after a new source arrives');
images[3].onload();
assert.equal(states[0].source_id,2);assert.equal(states[4].length,0);
second.onmessage({data:JSON.stringify({mode:'simulation',source_id:2,frame_id:2,preview_jpeg:'late'})});
cleanup();images[4].onload();
assert.equal(timers.size, 0);
console.log('PASS: source isolation, reconnect, decode before publish, bounded latest-frame queue, stale source rejection.');
