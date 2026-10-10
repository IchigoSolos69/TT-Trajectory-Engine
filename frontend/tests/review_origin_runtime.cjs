'use strict';
// Empty POST bodies cannot identify a clip or write annotations. This probe can
// safely characterize the compiled standalone guard before and after a build.
const { spawn } = require('node:child_process');
const { once } = require('node:events');
const fs = require('node:fs'), path = require('node:path'), net = require('node:net'), assert = require('node:assert/strict');
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function main() {
  const socket = net.createServer(); socket.listen(0, '127.0.0.1'); await once(socket, 'listening');
  const port = socket.address().port; await new Promise(resolve => socket.close(resolve));
  const child = spawn(process.execPath, ['server.js'], {
    cwd: path.resolve('frontend/.next-desktop/standalone'),
    env: { ...process.env, HOSTNAME: '127.0.0.1', PORT: String(port) },
    windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'],
  });
  let log = ''; child.stdout.on('data', data => log += data); child.stderr.on('data', data => log += data);
  child.on('error', error => log += error.stack);
  const expectedOld = process.argv.includes('--expect-old');
  const numericOrigin = `http://127.0.0.1:${port}`, aliasOrigin = `http://localhost:${port}`;
  try {
    for (let i = 0; i < 150 && !log.includes('Ready'); i++) await delay(100);
    assert(log.includes('Ready'), log);
    const post = async origin => {
      const response = await fetch(numericOrigin + '/api/training-review', {
        method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/json' }, body: '{}',
      });
      return { origin, status: response.status, body: await response.json() };
    };
    const requests = [];
    for (const origin of [numericOrigin, aliasOrigin, 'https://untrusted.example']) requests.push(await post(origin));
    const report = { expectedOld, requestOrigin: numericOrigin, requests, log };
    fs.writeFileSync(path.resolve('artifacts', expectedOld ? 'review_origin_before.json' : 'review_origin_after.json'), JSON.stringify(report, null, 2));
    assert.equal(requests[0].status, expectedOld ? 403 : 400, 'Exact numeric origin must pass the origin check after the fix (empty body rejected later)');
    assert.equal(requests[1].status, expectedOld ? 400 : 403, 'Loopback alias must remain a different origin after the fix');
    assert.equal(requests[2].status, 403);
    console.log('PASS: standalone compiled API ' + (expectedOld ? 'reproduces the old IP-origin rejection and alias acceptance.' : 'accepts its exact IP origin and rejects aliases/remote origins.'));
  } finally {
    if (child.exitCode === null) { const exit = once(child, 'exit'); child.kill(); await exit; }
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
