
import React from 'react';

interface LoaderProps {
  text: string;
}

export const Loader: React.FC<LoaderProps> = ({ text }) => {
  return (
    <div className="w-full h-full flex flex-col items-center justify-center text-center p-8 text-[#E5E5E5] bg-black">
      <p className="text-xl font-black mb-2 animate-pulse">{text}</p>
      <div className="w-1/2 h-1 bg-[#E5E5E5] mt-4 overflow-hidden">
        <div className="h-full bg-yellow-400 w-full animate-[loader-progress_2s_ease-in-out_infinite]"></div>
      </div>
      <style>{`
        @keyframes loader-progress {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </div>
  );
};
