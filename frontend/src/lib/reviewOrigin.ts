import type { NextRequest } from 'next/server';

const loopbackHosts = new Set(['localhost', '127.0.0.1', '[::1]']);

/** The direct HTTP authority preserves the browser's origin; NextURL rewrites
 * loopback IP addresses to localhost. Forwarded headers are never trusted here.
 */
export function localReviewOrigin(req: NextRequest): string {
  const authority = req.headers.get('host');
  if (!authority || /[\s/@?#\\]/.test(authority) || !loopbackHosts.has(req.nextUrl.hostname)) {
    throw Error('Training review is available on localhost only');
  }
  let address: URL;
  try { address = new URL(`${req.nextUrl.protocol}//${authority}`); }
  catch { throw Error('Training review is available on localhost only'); }
  if (!['http:', 'https:'].includes(address.protocol) || !loopbackHosts.has(address.hostname) ||
      address.port !== req.nextUrl.port || address.username || address.password || address.pathname !== '/') {
    throw Error('Training review is available on localhost only');
  }
  return address.origin;
}

export function sameOriginReview(req: NextRequest): boolean {
  const expected = localReviewOrigin(req);
  // Require the exact browser origin, including its port. Different loopback
  // spellings are different origins and must not bypass the CSRF check.
  return req.headers.get('origin') === expected;
}
