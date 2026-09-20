import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Table Tennis Trajectory Engine | TensorRT & Next.js Analytics',
  description: 'High-speed ping pong ball tracking, 120 FPS WebSocket telemetry, and Magnus spin physics sandbox built with C++ TensorRT and Next.js.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-[#07090e] text-slate-100 antialiased selection:bg-cyan-500 selection:text-slate-950">
        {children}
      </body>
    </html>
  );
}
