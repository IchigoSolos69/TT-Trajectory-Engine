import { PhysicsParams } from '../types/telemetry';

export interface FlightPoint {
  x: number; y: number; z: number;
  vx: number; vy: number; vz: number; time: number;
  isBounce?: boolean; bounceNumber?: number; isNetHit?: boolean;
}

/** Semi-implicit integration with swept surface contact, in SI units.
 * x is table length, y is height above floor, z is lateral displacement.
 * Spin and friction are illustrative material parameters, not measured spin.
 */
export function simulateFlight(p: PhysicsParams, dt = .001) {
  const radius = .02, surface = .76 + radius, netX = 1.87;
  const netTop = .9125, halfWidth = .7625;
  const force = .5 * 1.225 * Math.PI * radius ** 2 / .0027;
  const theta = p.launch_angle_deg * Math.PI / 180;
  const phi = p.side_angle_deg * Math.PI / 180;
  const speed = p.speed_kmh / 3.6;
  let vx = speed * Math.cos(theta) * Math.cos(phi), vy = speed * Math.sin(theta);
  let vz = speed * Math.cos(theta) * Math.sin(phi), x = 0, y = .95, z = 0, time = 0;
  const spin = p.spin_rpm * Math.PI / 30;
  let wy = p.spin_type === 'sidespin_left' ? spin : p.spin_type === 'sidespin_right' ? -spin : 0;
  let wz = p.spin_type === 'topspin' ? -spin : p.spin_type === 'backspin' ? spin : 0;
  const points: FlightPoint[] = [{ x, y, z, vx, vy, vz, time }];
  const bounces: FlightPoint[] = [];
  let net = { checked: false, hit: false, heightAtNet: 0, clearanceCm: 0,
    hitPoint: undefined as { x: number; y: number } | undefined };
  let maxHeight = y;
  for (let step = 0; step < Math.ceil(4 / dt); step++) {
    const v = Math.hypot(vx, vy, vz);
    const oldX = x, oldY = y, oldZ = z;
    const ax = force * (-p.drag_coefficient * v * vx + p.magnus_coefficient * radius * (wy * vz - wz * vy));
    const ay = -9.81 + force * (-p.drag_coefficient * v * vy + p.magnus_coefficient * radius * wz * vx);
    const az = force * (-p.drag_coefficient * v * vz - p.magnus_coefficient * radius * wy * vx);
    vx += ax * dt; vy += ay * dt; vz += az * dt;
    x += vx * dt; y += vy * dt; z += vz * dt; time += dt;
    // Air spin decay; retained through every contact.
    wy *= Math.exp(-.15 * dt); wz *= Math.exp(-.15 * dt);
    const contactX = netX - radius;
    if (!net.checked && oldX < contactX && x >= contactX) {
      const f = (contactX - oldX) / (x - oldX);
      const cy = oldY + f * (y - oldY), cz = oldZ + f * (z - oldZ);
      if (Math.abs(cz) <= halfWidth + .1525 + radius) {
        const hit = cy - radius <= netTop && cy + radius >= .76;
        net = { checked: true, hit, heightAtNet: cy, clearanceCm: (cy - radius - netTop) * 100,
          hitPoint: hit ? { x: contactX, y: cy } : undefined };
        if (hit) {
          x = contactX; y = Math.abs(cz) <= halfWidth ? Math.max(surface, cy) : cy;
          z = cz; vx = -Math.abs(vx) * .12; vy *= .5; vz *= .5;
          points.push({ x, y, z, vx, vy, vz, time, isNetHit: true });
        }
      }
    }
    if (vy < 0 && oldY >= surface && y <= surface) {
      const f = (oldY - surface) / (oldY - y);
      const cx = oldX + f * (x - oldX), cz = oldZ + f * (z - oldZ);
      if (cx >= .5 && cx <= 3.24 && Math.abs(cz) <= halfWidth) {
        const normalSpeed = -vy;
        vy = normalSpeed * p.table_restitution;
        // Coulomb impulse at the bottom of a hollow sphere (I = 2/3 mr²).
        const slipX = vx + radius * wz, slipZ = vz;
        const slip = Math.hypot(slipX, slipZ);
        const impulse = Math.min(slip / 2.5, p.table_friction * (1 + p.table_restitution) * normalSpeed);
        if (slip > 1e-8) {
          const jx = -impulse * slipX / slip, jz = -impulse * slipZ / slip;
          vx += jx; vz += jz; wz += 1.5 * jx / radius;
        }
        const contact: FlightPoint = { x: cx, y: surface, z: cz, vx, vy, vz,
          time: time - dt + f * dt, isBounce: true, bounceNumber: bounces.length + 1 };
        bounces.push(contact); points.push(contact);
        const remaining = (1 - f) * dt;
        x = cx + vx * remaining; y = surface + vy * remaining; z = cz + vz * remaining;
        if (normalSpeed < .15 || bounces.length >= 8) break;
      }
    }
    maxHeight = Math.max(maxHeight, y);
    if (y <= radius) {
      y = radius;
      points.push({ x, y, z, vx, vy, vz, time });
      break;
    }
    points.push({ x, y, z, vx, vy, vz, time });
    if (x < -1 || x > 5 || Math.abs(z) > 2.5) break;
  }
  return { points, bounces, net, maxHeight, flightTime: time };
}
