# Training review origin fix — 2026-10-09

The desktop training-review save error came from Next.js 14.2.35 normalizing `127.0.0.1` and `[::1]` to `localhost` in `NextRequest.nextUrl`. Chromium correctly sent the application's real origin, `http://127.0.0.1:<dynamic port>`, so comparison with `req.nextUrl.origin` rejected legitimate saves.

The route now obtains the browser-facing origin from the direct HTTP `Host` authority after validating a loopback request URL, an exact allowed loopback hostname, the request protocol and the server port. It still requires the exact `Origin` header. Loopback aliases and different ports remain different origins. Forwarded host headers are ignored. Path confinement, normalized annotation bounds, visibility consistency, optimistic revisions, immutable history and class-scoped review behavior remain enforced.

## Evidence

- [Before-fix standalone requests](artifacts/review_origin_before.json): the numeric-loopback origin received HTTP 403, while its mismatched `localhost` alias incorrectly passed the origin guard. Empty request bodies prevented any annotation write.
- [After-fix standalone requests](artifacts/review_origin_after.json): the numeric-loopback origin passed the guard and received the expected empty-body validation error; the alias and remote origin received HTTP 403.
- [Standalone review regression](artifacts/desktop_training_review_api.json): genuine reviewed fixture saves succeed; missing/remote/alias/cross-port origins and forwarded-host spoofing cannot authorize saves. Frame/path, label/landmark/tag, stale-revision, history and class-scoped persistence checks pass.
- [Actual NextRequest unit test](artifacts/review_origin_unit.log): reproduces the framework normalization and checks IPv6, exact ports, unsafe authorities and distinct loopback aliases.
- [Desktop production build](artifacts/blurball_desktop_frontend_build.log): compilation and TypeScript checks pass. Only `.next-desktop` was rebuilt; the website's `.next` output was preserved.
- Existing review annotations `training_data/live/phone_e23544b0c4c1a850_1/annotations.json` retain SHA-256 `2d561e8c9c53aed3a491d27e8a66115075011c5483f24345763cdcb44f6639df`.

The [final packaged desktop smoke](artifacts/blurball_desktop_smoke_final.json) also passed an actual Chromium renderer fetch save/reload with HTTP 200, with the browser supplying its own Origin header. It includes actual pointer streak review/save and authoritative flight-display checks and uses only the smoke runner's temporary data root. The verified release now occupies `desktop/dist/TT Trajectory Engine`; the prior build is retained for rollback.

## Re-run

From the project root, after building the desktop frontend:

```powershell
node frontend/tests/review_origin.test.cjs
node frontend/tests/review_origin_runtime.cjs
node frontend/tests/training_review_api.cjs --desktop
```

The runtime tests create temporary loopback services, stop their own services afterward, and keep all fixture annotation writes under `artifacts`.
