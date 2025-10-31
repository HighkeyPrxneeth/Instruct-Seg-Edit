
import React, { useCallback } from 'react';

interface ImageUploaderProps {
  onImageUpload: (image: HTMLImageElement) => void;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({ onImageUpload }) => {
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          onImageUpload(img);
        };
        img.src = e.target?.result as string;
      };
      reader.readAsDataURL(file);
    }
  };

  const onDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.currentTarget.classList.remove('border-yellow-400', 'bg-neutral-800');
    const file = event.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => onImageUpload(img);
            img.src = e.target?.result as string;
        };
        reader.readAsDataURL(file);
    }
  }, [onImageUpload]);

  const onDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.currentTarget.classList.add('border-yellow-400', 'bg-neutral-800');
  };
  
  const onDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.currentTarget.classList.remove('border-yellow-400', 'bg-neutral-800');
  };

  return (
    <div 
      className="w-full h-full flex flex-col items-center justify-center text-center p-8 text-[#E5E5E5] transition-colors duration-200"
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
    >
      <p className="text-xl font-black mb-2">DRAG AND DROP AN IMAGE</p>
      <p className="text-sm font-mono mb-6">OR</p>
      <label className="bg-yellow-400 text-black py-3 px-8 text-sm font-bold tracking-wider hover:bg-white focus:outline-none cursor-pointer transition-colors duration-200">
        SELECT FILE
        <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
      </label>
      <p className="text-xs mt-8 text-neutral-500 max-w-sm">YOUR IMAGE WILL BE PROCESSED LOCALLY AND UPLOADED FOR AI ANALYSIS. PLEASE USE IMAGES YOU HAVE THE RIGHTS TO.</p>
    </div>
  );
};
