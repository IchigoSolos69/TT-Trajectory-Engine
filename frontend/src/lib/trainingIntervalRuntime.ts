import fs from 'node:fs/promises';
import path from 'node:path';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
const execute=promisify(execFile);

export function intervalPreparer(){return process.env.TT_INTERVAL_PREPARER?path.resolve(process.env.TT_INTERVAL_PREPARER):path.resolve(process.cwd(),'..','python_tools','prepare_absence_interval.py');}
/** Training tools are external; native inference and review never need Python. */
export async function probeIntervalRuntime(){
  const worker=intervalPreparer();
  try{if(!(await fs.stat(worker)).isFile())throw Error('not a file');}catch{return {available:false as const,reason:'The full-interval training worker is missing. Use an updated local application package or the project training tools.'};}
  const candidates:string[]=[];
  const configured=process.env.TT_TRAINING_PYTHON??process.env.TT_PYTHON;
  if(configured){if(!path.isAbsolute(configured))return {available:false as const,reason:'TT_TRAINING_PYTHON must name an absolute local Python executable.'};candidates.push(configured);}
  else{
    if(process.env.LOCALAPPDATA)candidates.push(path.join(process.env.LOCALAPPDATA,'Microsoft','WindowsApps','python.exe'));
    if(process.platform==='win32'){
      try{
        const {stdout}=await execute(path.join(process.env.SystemRoot??'C:\\Windows','System32','where.exe'),['python.exe'],{windowsHide:true,timeout:5000,maxBuffer:100000});
        for(const entry of stdout.split(/\r?\n/).filter(Boolean))if(path.isAbsolute(entry)&&/[/\\](?:Microsoft[/\\]WindowsApps|Programs[/\\]Python|Python[0-9]+)[/\\]/i.test(entry)&&entry.toLowerCase().endsWith('python.exe'))candidates.push(entry);
      }catch{/* A missing PATH Python is a normal optional training dependency. */}
    }
  }
  const errors:string[]=[];
  for(const executable of Array.from(new Set(candidates))){
    try{
      const {stdout}=await execute(executable,['-B',worker,'--probe'],{windowsHide:true,timeout:15000,maxBuffer:1000000});
      const probe=JSON.parse(stdout.trim());if(probe.available!==true)throw Error(probe.reason??'OpenCV decoder probe did not pass');
      return {available:true as const,executable,worker,probe};
    }catch(error){errors.push(error instanceof Error?error.message:String(error));if(configured)break;}
  }
  return {available:false as const,reason:'Full-interval preparation needs a local Python installation with OpenCV. Set TT_TRAINING_PYTHON to its absolute executable, then check the decoder again. No software was installed.',diagnostics:errors};
}
