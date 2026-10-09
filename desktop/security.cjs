const path = require('node:path');
const { pathToFileURL } = require('node:url');
function allowedPage(url, origin, launcherFile) {
  try {
    const parsed = new URL(url);
    return (origin && parsed.origin === origin && ['/','/training-review'].includes(parsed.pathname)) || url === pathToFileURL(path.resolve(launcherFile)).href;
  } catch { return false; }
}
module.exports = { allowedPage };
