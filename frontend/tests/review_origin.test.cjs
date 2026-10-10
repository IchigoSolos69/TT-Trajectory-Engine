'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const ts = require('../node_modules/typescript');
const { NextRequest } = require('../node_modules/next/server');
const scope = { exports: {}, URL, Set, Error };
vm.runInNewContext(ts.transpileModule(fs.readFileSync(path.join(__dirname, '../src/lib/reviewOrigin.ts'), 'utf8'),
  { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText, scope);
const { localReviewOrigin, sameOriginReview } = scope.exports;
const request = (host, origin, extra = {}, internal = 'http://127.0.0.1:3100') => new NextRequest(internal + '/api/training-review', {
  method: 'POST', headers: { host, ...(origin === undefined ? {} : { origin }), ...extra }, body: '{}',
});

const desktop = request('127.0.0.1:3100', 'http://127.0.0.1:3100');
assert.equal(desktop.nextUrl.origin, 'http://localhost:3100', 'Reproduce NextURL IP normalization');
assert.notEqual(desktop.headers.get('origin'), desktop.nextUrl.origin, 'Old guard rejected Electron origin');
assert.equal(localReviewOrigin(desktop), 'http://127.0.0.1:3100');
assert.equal(sameOriginReview(desktop), true);
for (const host of ['localhost:3100', '127.0.0.1:3100', '[::1]:3100']) {
  assert.equal(sameOriginReview(request(host, 'http://' + host)), true);
  for (const origin of [undefined, 'null', 'https://untrusted.example', 'http://127.0.0.1:3101', 'https://127.0.0.1:3100', 'http://' + host + '/', 'http://' + host + ',http://untrusted.example']) {
    assert.equal(sameOriginReview(request(host, origin)), false, `${host} must reject ${origin}`);
  }
}
assert.equal(sameOriginReview(request('127.0.0.1:3100', 'http://localhost:3100')), false, 'Loopback aliases remain separate origins');
assert.equal(sameOriginReview(request('localhost:3100', 'http://127.0.0.1:3100')), false);
assert.throws(() => localReviewOrigin(request('localhost:3100', 'http://localhost:3100', {}, 'http://untrusted.example:3100')), /localhost only/);
for (const host of ['untrusted.example:3100', '127.0.0.2:3100', 'localhost.:3100', 'localhost:3101', 'localhost:3100/path', 'user@localhost:3100', 'localhost:3100,untrusted.example', '', '[::2]:3100']) {
  assert.throws(() => localReviewOrigin(request(host, 'http://' + host)), /localhost only/, `Reject authority ${host}`);
}
assert.equal(sameOriginReview(request('127.0.0.1:3100', 'http://127.0.0.1:3100', { 'x-forwarded-host': 'untrusted.example', 'forwarded': 'host=untrusted.example' })), true);
assert.equal(sameOriginReview(request('127.0.0.1:3100', 'https://untrusted.example', { 'x-forwarded-host': 'untrusted.example' })), false);
assert.equal(sameOriginReview(request('127.0.0.1:3100', 'http://127.0.0.1:3100', { 'x-forwarded-host': 'localhost:3100' })), true);
console.log('PASS: actual NextRequest normalization reproduced; exact direct loopback origin accepted; port, alias, authority and forwarded-host attacks rejected.');
