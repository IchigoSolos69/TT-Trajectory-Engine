'use client';

import React, { useEffect, useRef } from 'react';
import gsap from 'gsap';

interface AnimatedNumberProps {
  value: number;
  decimals?: number;
  suffix?: string;
  className?: string;
}

export const AnimatedNumber: React.FC<AnimatedNumberProps> = ({
  value,
  decimals = 1,
  suffix = '',
  className = 'font-mono font-bold text-emerald-400 drop-shadow-[0_0_8px_rgba(16,185,129,0.5)]',
}) => {
  const numberRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!numberRef.current) return;

    // Parse current text to start the animation from the existing number
    const startValue = parseFloat(numberRef.current.innerText) || 0;
    const proxy = { val: startValue };

    gsap.to(proxy, {
      val: value,
      duration: 0.8,
      ease: 'power3.out', // Fast start, smooth deceleration
      onUpdate: () => {
        if (numberRef.current) {
          numberRef.current.innerText = `${proxy.val.toFixed(decimals)}${suffix}`;
        }
      },
    });
  }, [value, decimals, suffix]);

  return (
    <span ref={numberRef} className={className}>
      {value.toFixed(decimals)}
      {suffix}
    </span>
  );
};

export default AnimatedNumber;
