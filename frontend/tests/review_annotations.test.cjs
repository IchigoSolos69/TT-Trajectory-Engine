const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const ts=require('../node_modules/typescript');
function load(name,dependencies={}){const scope={exports:{},require:name=>dependencies[name]};vm.runInNewContext(ts.transpileModule(fs.readFileSync(path.join(__dirname,'../src/lib',name),'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS}}).outputText,scope);return scope.exports;}
const geometry=load('reviewGeometry.ts'),review=load('reviewAnnotations.ts',{'./reviewGeometry':geometry});
const label={reviewed:true,reviewed_classes:{ball:true,person:false,table:false},ball:[[.4,.5,.01,.01]],person:[],table:[],geometry:{...geometry.emptyGeometry(),visibility:'visible'}};
assert(review.classReviewed(label,'ball'));assert(!review.classReviewed(label,'person'));
assert(review.classReviewed({reviewed:true},'table'),'Legacy full review remains recognized');
assert(review.ballReviewProblem({...label,ball:[]}).includes('no box'));
assert(review.ballReviewProblem({...label,geometry:{...label.geometry,visibility:'occluded'}}).includes('hidden'));
assert(review.ballReviewProblem({...label,ball:[],geometry:geometry.emptyGeometry()}).includes('cannot tell'));
const frames=[{file:'frame_1.jpg',source_frame:1,burst_id:1},{file:'frame_2.jpg',source_frame:2,burst_id:1}];
const copied=review.previousBallSuggestion(frames,1,{'frame_1.jpg':label});
assert.equal(JSON.stringify(copied),JSON.stringify(label.ball[0]));copied[0]=0;assert.equal(label.ball[0][0],.4,'Suggestions cannot mutate saved labels');
for(const changed of [{...frames[1],source_frame:5},{...frames[1],burst_id:2},{...frames[1],burst_id:undefined}])assert.equal(review.previousBallSuggestion([frames[0],changed],1,{'frame_1.jpg':label}),null);
assert.equal(review.previousBallSuggestion(frames,1,{'frame_1.jpg':{...label,reviewed_classes:{ball:false,person:true,table:true}}}),null);
assert.equal(review.nextBallReviewIndex(frames,1,{'frame_1.jpg':{...label,ball:[]},'frame_2.jpg':label}),0,'Review queue wraps to an earlier conflicting label');
assert.equal(review.nextBallReviewIndex(frames,1,{'frame_1.jpg':label,'frame_2.jpg':label}),-1);
assert(review.validDraft(label));assert(!review.validDraft({...label,ball:[[.99,.5,.5,.1]]}));assert(!review.validDraft({...label,geometry:{visibility:'visible'}}));
console.log('PASS: class-scoped review, missing-ball warning, bounded box reuse and recovered draft validation.');
