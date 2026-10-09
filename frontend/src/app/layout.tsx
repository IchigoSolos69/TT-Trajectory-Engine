import type { Metadata } from 'next';
import './globals.css';
import {DesktopControls} from '../components/DesktopControls';

export const metadata: Metadata = {
  title: 'Table Tennis Trajectory Engine | TensorRT & Next.js Analytics',
  description: 'Table tennis footage analysis with TensorRT ball and scene detection, measured WebSocket telemetry, and a Magnus spin physics sandbox.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-[#07090e] text-slate-100 antialiased selection:bg-cyan-500 selection:text-slate-950">
        <DesktopControls />
        {children}
      </body>
    </html>
  );
}
