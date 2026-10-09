/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',
  distDir: process.env.TT_DESKTOP_BUILD === '1' ? '.next-desktop' : '.next',
}

module.exports = nextConfig
