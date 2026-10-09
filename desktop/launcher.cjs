'use strict';
const status = document.getElementById('status');
const buttons = [...document.querySelectorAll('button')];
function update(context) {
  buttons.forEach(button => button.disabled = context.phase === 'starting');
  status.classList.toggle('error', context.phase === 'error');
  status.textContent = context.error || (context.phase === 'starting' ? 'Preparing your local workspace… Initial engine loading may take a moment.' : 'Ready. Choose a recording or a live session.');
}
async function action(run) {
  try { await run(); } catch (error) { update({phase:'error',error:error.message}); }
}
document.getElementById('recording').onclick = () => action(() => window.ttDesktop.openRecording());
document.getElementById('live').onclick = () => action(() => window.ttDesktop.startLive());
document.getElementById('data').onclick = () => action(() => window.ttDesktop.showDataDirectory());
window.ttDesktop.onSessionChanged(update);
window.ttDesktop.getContext().then(update).catch(error => update({phase:'error',error:error.message}));
