// A health route alone stays responsive when a running Next server references
// deleted chunks from a previous build. Verify the assets in its actual HTML.
async function checkFrontend(baseUrl) {
  const base = new URL(baseUrl);
  const request = url => fetch(url, { signal: AbortSignal.timeout(4000), cache: 'no-store', redirect: 'error' });
  const identityResponse = await request(new URL('/api/health', base));
  if (!identityResponse.ok) throw Error(`Dashboard identity HTTP ${identityResponse.status}`);
  const identity = await identityResponse.json();
  if (identity.application !== 'TT_TRAJECTORY_DASHBOARD') throw Error('Dashboard identity mismatch');
  const page = await request(new URL('/', base));
  if (!page.ok) throw Error(`Dashboard page HTTP ${page.status}`);
  const html = await page.text();
  const assets = [...new Set([...html.matchAll(/(?:src|href)="(\/_next\/static\/[^"<>]+)"/g)].map(match => match[1]))];
  if (!assets.some(asset => asset.endsWith('.css')) || !assets.some(asset => asset.endsWith('.js'))) {
    throw Error('Dashboard HTML has no CSS/JavaScript build assets');
  }
  const checks = await Promise.allSettled(assets.map(async asset => {
    const url = new URL(asset, base);
    if (url.origin !== base.origin) throw Error('Dashboard asset origin mismatch');
    const response = await fetch(url, { method: 'HEAD', signal: AbortSignal.timeout(4000), cache: 'no-store', redirect: 'error' });
    const expected = asset.endsWith('.css') ? /text\/css/i : /(?:application|text)\/javascript/i;
    if (!response.ok || !expected.test(response.headers.get('content-type') || '')) {
      throw Error(`Dashboard build asset unavailable: ${asset} (HTTP ${response.status}). Stop the project frontend, rebuild, then restart it.`);
    }
  }));
  // Drain every request before reporting a failure or exiting the probe.
  const failure = checks.find(check => check.status === 'rejected');
  if (failure) throw failure.reason;
  return { ...identity, assets_checked: assets.length };
}
module.exports = { checkFrontend };
