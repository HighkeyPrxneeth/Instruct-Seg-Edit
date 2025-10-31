
import React, { useState, useEffect } from 'react';
import { Loader } from './Loader';

interface DiffusionViewerProps {
  finalImage: string | null;
}

const STEPS = 8;
const STEP_DURATION = 150; // ms

export const DiffusionViewer: React.FC<DiffusionViewerProps> = ({ finalImage }) => {
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    if (finalImage) {
      setCurrentStep(0);
      const interval = setInterval(() => {
        setCurrentStep(prev => {
          if (prev >= STEPS) {
            clearInterval(interval);
            return STEPS;
          }
          return prev + 1;
        });
      }, STEP_DURATION);
      return () => clearInterval(interval);
    }
  }, [finalImage]);

  if (!finalImage) {
    return <Loader text="INITIALIZING GENERATION" />;
  }

  const opacity = (currentStep / STEPS);

  return (
    <div className="w-full h-full relative bg-black flex items-center justify-center overflow-hidden">
      {/* Noise background */}
      <svg className="absolute inset-0 w-full h-full" >
        <filter id="noise">
          <feTurbulence type="fractalNoise" baseFrequency="0.8" numOctaves="4" stitchTiles="stitch"/>
        </filter>
        <rect width="100%" height="100%" filter="url(#noise)" opacity="0.2"/>
      </svg>

      <div className="relative w-full h-full">
        <img
          src={finalImage}
          alt="Generating"
          className="absolute inset-0 w-full h-full object-contain transition-opacity duration-500 ease-in-out"
          style={{ opacity: opacity, filter: `blur(${Math.max(0, 16 - opacity * 16)}px)` }}
        />
      </div>

      <div className="absolute bottom-4 left-4 bg-black/50 text-white p-2 font-mono text-xs">
        <p>DIFFUSION STEP: {currentStep}/{STEPS}</p>
        <p>STATUS: {currentStep < STEPS ? 'REFINING...' : 'COMPLETE'}</p>
      </div>
    </div>
  );
};
