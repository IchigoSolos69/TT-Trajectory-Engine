import path from 'node:path';
export const dynamic = 'force-dynamic';
export async function GET() {
  return Response.json({application:'TT_TRAJECTORY_DASHBOARD',pid:process.pid,root:process.env.TT_PROJECT_ROOT?path.resolve(process.env.TT_PROJECT_ROOT):path.resolve(process.cwd(),'..')});
}
