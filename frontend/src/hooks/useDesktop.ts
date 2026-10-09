'use client';
import {useEffect,useState} from 'react';
import {DesktopContext} from '../types/desktop';
export function useDesktop() {
  const [context,setContext]=useState<DesktopContext|null>(null);
  const [initialized,setInitialized]=useState(false);
  useEffect(()=>{
    const api=window.ttDesktop;
    if(!api){setInitialized(true);return;}
    let active=true;
    const update=(value:DesktopContext)=>{if(active){setContext(value);setInitialized(true);}};
    const remove=api.onSessionChanged(update);
    api.getContext().then(update).catch(error=>update({phase:'error',backendUrl:null,mode:null,error:String(error)}));
    return()=>{active=false;remove();};
  },[]);
  return {context,initialized};
}
