'use strict';
const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('ttDesktop', {
  getContext: () => ipcRenderer.invoke('desktop:context'),
  openRecording: () => ipcRenderer.invoke('desktop:recording'),
  startLive: () => ipcRenderer.invoke('desktop:live'),
  endSession: () => ipcRenderer.invoke('desktop:end'),
  showDataDirectory: () => ipcRenderer.invoke('desktop:data'),
  getReviewDrafts: () => ipcRenderer.invoke('desktop:drafts-read'),
  saveReviewDrafts: value => ipcRenderer.invoke('desktop:drafts-save',value),
  onSessionChanged: callback => {
    const listener = (_event, context) => callback(context);
    ipcRenderer.on('desktop:session', listener);
    return () => ipcRenderer.removeListener('desktop:session', listener);
  },
});
