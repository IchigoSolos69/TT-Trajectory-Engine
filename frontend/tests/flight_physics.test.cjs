const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const ts = require('../node_modules/typescript');
const context = { exports: {} };
vm.runInNewContext(ts.transpileModule(fs.readFileSync(require('node:path').join(__dirname,'../src/lib/flightPhysics.ts'),'utf8'),
  { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText, context);
const simulate = context.exports.simulateFlight;
const base = {speed_kmh:20,launch_angle_deg:12,side_angle_deg:0,spin_rpm:4800,spin_type:'topspin',drag_coefficient:.4,table_restitution:.89,magnus_coefficient:.28,table_friction:.22};
let repeated;
for (const speed of [8,10,12,14,16,18,20,22,24]) {
  for (const angle of [0,5,10,15]) {
    const p={...base,speed_kmh:speed,launch_angle_deg:angle};
    const result=simulate(p);
    if(result.bounces.length>=2 && !result.net.hit) repeated={p,result};
    assert(result.points.every(p=>Number.isFinite(p.x+p.y+p.z+p.time)), 'finite integration');
    for (const pt of result.points) {
      if(result.bounces.length && pt.time>=result.bounces[0].time && pt.x>=.5 && pt.x<=3.24 && Math.abs(pt.z)<=.7625) assert(pt.y>=.779999,'no tunneling through tabletop after contact');
    }
    result.bounces.forEach((pt,i)=>{
      assert.equal(pt.bounceNumber,i+1); assert.equal(pt.y,.78); assert(pt.vy>0);
      if(i) assert(pt.time>result.bounces[i-1].time,'distinct chronological contacts');
    });
  }
}
assert(repeated,'at least one physically valid second table bounce');
const coarse=simulate(repeated.p,.001),fine=simulate(repeated.p,.0005);
assert(Math.abs(coarse.bounces[1].x-fine.bounces[1].x)<.025,'second bounce converges within 2.5 cm');
const noSpin=simulate({...base,spin_rpm:0,magnus_coefficient:0,table_friction:0,drag_coefficient:0,speed_kmh:12,launch_angle_deg:0});
const b=noSpin.bounces[0]; assert(b,'first bounce exists');
const expected=Math.sqrt(2*9.81*(.95-.78))*.89;
assert(Math.abs(b.vy-expected)<.02,'restitution matches analytical drop');
const sideways=simulate({...base,speed_kmh:30,side_angle_deg:75});
assert.equal(sideways.bounces.length,0,'off-table ball must fall to floor');
assert.equal(sideways.net.checked,false,'no invented net clearance for a miss');
console.log(JSON.stringify({passed:true,sweep_cases:36,second_bounce_example:repeated.p,second_bounce:coarse.bounces[1],second_bounce_convergence_m:Math.abs(coarse.bounces[1].x-fine.bounces[1].x),restitution_error_ms:Math.abs(b.vy-expected)},null,2));
