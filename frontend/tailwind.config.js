/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f0fdf4',
          500: '#22c55e',
          600: '#16a34a',
          900: '#14532d',
        },
        cyanNeon: '#00f3ff',
        orangeNeon: '#ff6600',
        tableGreen: '#0b5345',
        tableBlue: '#1b4f72',
      },
      fontFamily: {
        sans: ['var(--font-outfit)', 'sans-serif'],
      },
      boxShadow: {
        glowCyan: '0 0 25px rgba(0, 243, 255, 0.4)',
        glowOrange: '0 0 25px rgba(255, 102, 0, 0.4)',
        glowGreen: '0 0 25px rgba(34, 197, 94, 0.4)',
      },
      animation: {
        pulseSlow: 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        radarSpin: 'radar 4s linear infinite',
      },
      keyframes: {
        radar: {
          '0%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' },
        }
      }
    },
  },
  plugins: [],
}
