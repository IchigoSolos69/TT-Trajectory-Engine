// Real HTTP acceptance check; no browser, camera or backend authorization.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { checkFrontend } = require('./frontend_health.cjs');
const root = path.resolve(__dirname, '..');
const base = `http://127.0.0.1:${Number(process.argv[2] || 3000)}`;
const hash = buffer => crypto.createHash('sha256').update(buffer).digest('hex');

async function main() {
  const health = await checkFrontend(base);
  assert.equal(path.resolve(health.root), root);
  const buildId = fs.readFileSync(path.join(root, 'frontend/.next/BUILD_ID'), 'utf8').trim();
  const pages = [];
  for (const route of ['/', '/training-review']) {
    const response = await fetch(base + route, { signal: AbortSignal.timeout(10000) });
    assert.equal(response.status, 200);
    const html = await response.text();
    assert.ok(html.includes(buildId), `${route} must reference the current disk build`);
    const urls = [...new Set([...html.matchAll(/(?:href|src)="(\/_next\/static\/[^"<>]+)"/g)].map(match => match[1]))];
    assert.ok(urls.some(url => url.endsWith('.css')) && urls.some(url => url.endsWith('.js')));
    const assets = [];
    for (const url of urls) {
      const asset = await fetch(base + url, { signal: AbortSignal.timeout(10000) });
      assert.equal(asset.status, 200, url);
      const body = Buffer.from(await asset.arrayBuffer());
      const disk = fs.readFileSync(path.join(root, 'frontend/.next', url.slice('/_next/'.length)));
      assert.ok(body.length > 0, url);
      assert.equal(hash(body), hash(disk), `${url} must match the current disk asset`);
      assets.push({ url, status: asset.status, bytes: body.length, sha256: hash(body) });
    }
    pages.push({ route, status: response.status, assets });
  }
  const icon = await fetch(base + '/favicon.ico');
  assert.equal(icon.status, 200);
  const iconBytes = Buffer.from(await icon.arrayBuffer());
  assert.equal(iconBytes.readUInt16LE(2), 1, 'ICO format');
  assert.equal(hash(iconBytes), hash(fs.readFileSync(path.join(root, 'frontend/public/favicon.ico'))));
  const report = { passed: true, checked_at: new Date().toISOString(), scope: 'HTTP page/build/asset integrity; not a browser render test', base, buildId, health, pages, favicon: { status: icon.status, bytes: iconBytes.length } };
  fs.writeFileSync(path.join(root, 'artifacts/frontend_asset_runtime.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(`PASS: current build ${buildId}; dashboard and review CSS/JS match disk; favicon HTTP 200`);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
