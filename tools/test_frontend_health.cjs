const assert = require('node:assert/strict');
const http = require('node:http');
const path = require('node:path');
const { promisify } = require('node:util');
const execFile = promisify(require('node:child_process').execFile);
const { checkFrontend } = require('./frontend_health.cjs');

async function main() {
  let assetStatus = 200, assetMime = 'application/javascript', identity = 'TT_TRAJECTORY_DASHBOARD';
  const server = http.createServer((req, res) => {
    if (req.url === '/api/health') {
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify({ application: identity, pid: process.pid }));
    } else if (req.url === '/') {
      res.setHeader('Content-Type', 'text/html');
      res.end('<link href="/_next/static/css/build.css" rel="stylesheet"><script src="/_next/static/chunks/app/page-build.js"></script>');
    } else {
      res.statusCode = req.url.endsWith('.js') ? assetStatus : 200;
      res.setHeader('Content-Type', req.url.endsWith('.css') ? 'text/css' : assetMime);
      res.end();
    }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const url = `http://127.0.0.1:${server.address().port}`;
    assert.equal((await checkFrontend(url)).assets_checked, 2);
    for (const status of [400, 404]) {
      assetStatus = status;
      await assert.rejects(checkFrontend(url), new RegExp(`build asset unavailable:.*HTTP ${status}`));
    }
    await assert.rejects(execFile(process.execPath, [path.join(__dirname, 'service_probe.cjs'), 'frontend', String(server.address().port)]), error => {
      assert.equal(error.code, 1);
      assert.match(error.stderr, /build asset unavailable/);
      assert.doesNotMatch(error.stderr, /Assertion failed/);
      return true;
    });
    assetStatus = 200; assetMime = 'text/html';
    await assert.rejects(checkFrontend(url), /build asset unavailable/);
    assetMime = 'application/javascript'; identity = 'OTHER_APPLICATION';
    await assert.rejects(checkFrontend(url), /identity mismatch/);
    console.log('PASS: valid frontend; stale/missing chunks, HTML asset responses and unrelated identity rejected; CLI exits cleanly on stale build');
  } finally {
    await new Promise(resolve => server.close(resolve));
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
