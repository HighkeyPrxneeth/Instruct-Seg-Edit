import React, {
  useRef,
  useEffect,
  useState,
  useImperativeHandle,
  forwardRef,
  useCallback,
} from 'react';
import type { BrushState, EditorCanvasHandle } from '../types';

interface EditorCanvasProps {
  image: HTMLImageElement;
  mask: string | null;
  brushState: BrushState;
  isEditable: boolean;
  finalImage: string | null;
  onHistoryChange: (state: { canUndo: boolean; canRedo: boolean }) => void;
}

const EditorCanvasComponent: React.ForwardRefRenderFunction<
  EditorCanvasHandle,
  EditorCanvasProps
> = (
  { image, mask, brushState, isEditable, finalImage, onHistoryChange },
  ref
) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const cursorRef = useRef<HTMLDivElement>(null);
  
  const [history, setHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);

  const [isDrawing, setIsDrawing] = useState(false);
  const lastPos = useRef<{ x: number; y: number } | null>(null);

  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  const getImageDrawProps = useCallback(() => {
    if (!image || !containerRef.current) return { x: 0, y: 0, width: 0, height: 0 };
    
    const { clientWidth, clientHeight } = containerRef.current;
    const imgAspectRatio = image.naturalWidth / image.naturalHeight;
    const containerAspectRatio = clientWidth / clientHeight;

    let drawWidth, drawHeight;
    if (imgAspectRatio > containerAspectRatio) {
      drawWidth = clientWidth;
      drawHeight = clientWidth / imgAspectRatio;
    } else {
      drawHeight = clientHeight;
      drawWidth = clientHeight * imgAspectRatio;
    }
    
    const x = (clientWidth - drawWidth) / 2;
    const y = (clientHeight - drawHeight) / 2;

    return { x, y, width: drawWidth, height: drawHeight };
  }, [image]);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!ctx || !image || dimensions.width === 0) return;
    
    const { x, y, width, height } = getImageDrawProps();
    
    ctx.clearRect(0, 0, canvas!.width, canvas!.height);
    ctx.drawImage(image, x, y, width, height);

    if (finalImage) {
        const finalImg = new Image();
        finalImg.onload = () => {
            ctx.drawImage(finalImg, x, y, width, height);
        };
        finalImg.src = finalImage;
    } else if (maskCanvasRef.current && historyIndex >= 0) {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = width;
        tempCanvas.height = height;
        const tempCtx = tempCanvas.getContext('2d');
        if (!tempCtx) return;
        
        tempCtx.drawImage(maskCanvasRef.current, 0, 0, width, height);
        
        tempCtx.globalCompositeOperation = 'source-in';
        tempCtx.fillStyle = 'rgba(236, 72, 153, 0.6)'; // Pink overlay
        tempCtx.fillRect(0, 0, width, height);

        ctx.drawImage(tempCanvas, x, y, width, height);
    }
  }, [image, finalImage, dimensions, historyIndex, getImageDrawProps]);

  useEffect(() => {
    if (!image || !containerRef.current) return;
    
    const { width, height } = getImageDrawProps();
    setDimensions({ width, height });
    
    if (!maskCanvasRef.current) {
        maskCanvasRef.current = document.createElement('canvas');
    }
    maskCanvasRef.current.width = image.naturalWidth;
    maskCanvasRef.current.height = image.naturalHeight;
    const maskCtx = maskCanvasRef.current.getContext('2d');
    if(maskCtx) {
        maskCtx.clearRect(0, 0, image.naturalWidth, image.naturalHeight);
    }
    
    setHistory([]);
    setHistoryIndex(-1);
  }, [image, getImageDrawProps]);

  useEffect(() => {
    if (mask && image && history.length === 0 && maskCanvasRef.current) {
      const maskImg = new Image();
      maskImg.crossOrigin = "anonymous";
      maskImg.onload = () => {
        const maskCtx = maskCanvasRef.current!.getContext('2d');
        if (!maskCtx) return;

        maskCtx.drawImage(maskImg, 0, 0, image.naturalWidth, image.naturalHeight);
        
        const imageData = maskCtx.getImageData(0, 0, image.naturalWidth, image.naturalHeight);
        const data = imageData.data;
        for (let i = 0; i < data.length; i += 4) {
          const isBlack = data[i] < 128;
          if (isBlack) {
            data[i + 3] = 0;
          } else {
            data[i] = 255;
            data[i+1] = 255;
            data[i+2] = 255;
            data[i + 3] = 255;
          }
        }
        maskCtx.putImageData(imageData, 0, 0);

        const initialMaskData = maskCanvasRef.current!.toDataURL();
        setHistory([initialMaskData]);
        setHistoryIndex(0);
      };
      maskImg.src = mask;
    }
  }, [mask, image, history.length]);

  useEffect(() => {
    if (historyIndex === -1 || !maskCanvasRef.current || !history[historyIndex]) {
        redraw();
        return;
    };
    
    const maskImg = new Image();
    maskImg.onload = () => {
      const maskCtx = maskCanvasRef.current!.getContext('2d');
      if (maskCtx) {
        maskCtx.clearRect(0, 0, maskCanvasRef.current!.width, maskCanvasRef.current!.height);
        maskCtx.drawImage(maskImg, 0, 0);
        redraw();
      }
    };
    maskImg.src = history[historyIndex];
  }, [historyIndex, history, redraw]);

  useEffect(() => {
    redraw();
  }, [image, finalImage, dimensions, redraw]);

  useImperativeHandle(ref, () => ({
    exportMask: () => {
        if (!maskCanvasRef.current) return null;
        
        const exportCanvas = document.createElement('canvas');
        exportCanvas.width = maskCanvasRef.current.width;
        exportCanvas.height = maskCanvasRef.current.height;
        const exportCtx = exportCanvas.getContext('2d');
        if (!exportCtx) return null;

        exportCtx.fillStyle = 'black';
        exportCtx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
        
        exportCtx.drawImage(maskCanvasRef.current, 0, 0);
        
        return exportCanvas.toDataURL('image/png');
    },
    undo: () => {
      if (historyIndex > 0) setHistoryIndex(historyIndex - 1);
    },
    redo: () => {
      if (historyIndex < history.length - 1) setHistoryIndex(historyIndex + 1);
    },
  }));

  useEffect(() => {
    onHistoryChange({
      canUndo: historyIndex > 0,
      canRedo: historyIndex < history.length - 1,
    });
  }, [history, historyIndex, onHistoryChange]);

  const getCanvasCoordinates = (event: React.MouseEvent | React.PointerEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const { x, y, width, height } = getImageDrawProps();
    const rect = canvas.getBoundingClientRect();
    
    const physicalX = event.clientX - rect.left;
    const physicalY = event.clientY - rect.top;

    if (physicalX < x || physicalX > x + width || physicalY < y || physicalY > y + height) {
      return null;
    }

    const scaleX = image.naturalWidth / width;
    const scaleY = image.naturalHeight / height;
    
    return {
      x: (physicalX - x) * scaleX,
      y: (physicalY - y) * scaleY,
    };
  };

  const draw = useCallback((startPos: {x: number, y: number}, endPos: {x: number, y: number}) => {
    const maskCtx = maskCanvasRef.current?.getContext('2d');
    if (!maskCtx || !isEditable) return;

    maskCtx.lineCap = 'round';
    maskCtx.lineJoin = 'round';
    
    const { width } = getImageDrawProps();
    maskCtx.lineWidth = brushState.size * (image.naturalWidth / width);

    if (brushState.mode === 'ADD') {
        maskCtx.globalCompositeOperation = 'source-over';
        maskCtx.strokeStyle = 'white';
    } else {
        maskCtx.globalCompositeOperation = 'destination-out';
    }
    
    maskCtx.beginPath();
    maskCtx.moveTo(startPos.x, startPos.y);
    maskCtx.lineTo(endPos.x, endPos.y);
    maskCtx.stroke();
    
    // Reset composite operation to default to avoid bugs
    maskCtx.globalCompositeOperation = 'source-over';

  }, [brushState, isEditable, image.naturalWidth, getImageDrawProps]);
  
  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isEditable || finalImage || !e.isPrimary) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    setIsDrawing(true);
    const pos = getCanvasCoordinates(e);
    if(pos) {
        lastPos.current = pos;
        draw(pos, pos);
        redraw();
    }
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (cursorRef.current) {
        const rect = e.currentTarget.getBoundingClientRect();
        cursorRef.current.style.left = `${e.clientX - rect.left}px`;
        cursorRef.current.style.top = `${e.clientY - rect.top}px`;
    }

    if (!isDrawing || !isEditable || finalImage || !e.isPrimary) return;
    const pos = getCanvasCoordinates(e);
    if (pos && lastPos.current) {
        draw(lastPos.current, pos);
        redraw();
        lastPos.current = pos;
    } else {
      lastPos.current = pos;
    }
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !e.isPrimary) return;
    e.currentTarget.releasePointerCapture(e.pointerId);
    setIsDrawing(false);
    lastPos.current = null;
    
    if (maskCanvasRef.current) {
      const newMaskData = maskCanvasRef.current.toDataURL();
      const newHistory = history.slice(0, historyIndex + 1);
      newHistory.push(newMaskData);
      setHistory(newHistory);
      setHistoryIndex(newHistory.length - 1);
    }
  };

  const handlePointerEnter = () => {
    if (cursorRef.current) cursorRef.current.style.display = 'block';
  }

  const handlePointerLeave = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (cursorRef.current) cursorRef.current.style.display = 'none';
    if(isDrawing) handlePointerUp(e);
  }
  
  // Sync cursor style with brush state
  useEffect(() => {
    if (cursorRef.current) {
        cursorRef.current.style.width = `${brushState.size}px`;
        cursorRef.current.style.height = `${brushState.size}px`;
        
        const isAdd = brushState.mode === 'ADD';
        cursorRef.current.classList.toggle('border-pink-500', isAdd);
        cursorRef.current.classList.toggle('bg-pink-500/30', isAdd);
        cursorRef.current.classList.toggle('border-white', !isAdd);
        cursorRef.current.classList.toggle('bg-white/30', !isAdd);
    }
  }, [brushState]);
  
  return (
    <div ref={containerRef} className="w-full h-full flex items-center justify-center touch-none relative">
        <canvas
            ref={canvasRef}
            width={containerRef.current?.clientWidth ?? 0}
            height={containerRef.current?.clientHeight ?? 0}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerEnter={handlePointerEnter}
            onPointerLeave={handlePointerLeave}
            className={`${isEditable && !finalImage ? 'cursor-none' : 'cursor-default'}`}
        />
        {isEditable && !finalImage && (
            <div
                ref={cursorRef}
                className="absolute rounded-full border-2 pointer-events-none hidden"
                style={{
                    transform: 'translate(-50%, -50%)',
                }}
            />
        )}
    </div>
  );
};

export const EditorCanvas = forwardRef(EditorCanvasComponent);