
import React, { useState, useCallback, useRef } from 'react';
import { ControlPanel } from './components/ControlPanel';
import { EditorCanvas } from './components/EditorCanvas';
import { DiffusionViewer } from './components/DiffusionViewer';
import { ImageUploader } from './components/ImageUploader';
import { generateMask, generateEdit } from './services/backendService';
import type { AppState, BrushState, EditorCanvasHandle } from './types';
import { Loader } from './components/Loader';

const App: React.FC = () => {
  const [appState, setAppState] = useState<AppState>('IDLE');
  const [originalImage, setOriginalImage] = useState<HTMLImageElement | null>(null);
  const [maskImage, setMaskImage] = useState<string | null>(null);
  const [editedImage, setEditedImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [segmentationPrompt, setSegmentationPrompt] = useState<string>('');
  const [editPrompt, setEditPrompt] = useState<string>('');
  
  const [brushState, setBrushState] = useState<BrushState>({ mode: 'ADD', size: 30 });
  const [inferenceSteps, setInferenceSteps] = useState<number>(30);
  const [seed, setSeed] = useState<number>(() => Math.floor(Math.random() * 10000));
  
  const [undoRedoState, setUndoRedoState] = useState({ canUndo: false, canRedo: false });

  const editorCanvasRef = useRef<EditorCanvasHandle>(null);

  const handleImageUpload = useCallback((image: HTMLImageElement) => {
    setOriginalImage(image);
    setMaskImage(null);
    setEditedImage(null);
    setError(null);
    setAppState('SEGMENT_PENDING');
  }, []);

  const handleSegment = useCallback(async () => {
    if (!originalImage || !segmentationPrompt) {
      setError('Original image and segmentation prompt are required.');
      return;
    }
    setAppState('SEGMENTING');
    setError(null);
    try {
      const base64Image = originalImage.src;
      const generatedMask = await generateMask(base64Image, segmentationPrompt);
      setMaskImage(generatedMask);
      setAppState('MASK_REVIEW');
    } catch (e) {
      setError('Failed to generate segmentation mask. Please try again.');
      setAppState('SEGMENT_PENDING');
      console.error(e);
    }
  }, [originalImage, segmentationPrompt]);

  const handleGenerate = useCallback(async () => {
    if (!originalImage || !editPrompt) {
      setError('Image, mask, and edit prompt are required.');
      return;
    }
    
    const currentMask = editorCanvasRef.current?.exportMask();
    if (!currentMask) {
      setError('Could not get the edited mask.');
      return;
    }

    setAppState('GENERATING');
    setError(null);
    try {
      const base64Image = originalImage.src;
      const generatedEditResult = await generateEdit(base64Image, currentMask, editPrompt, inferenceSteps, seed);
      setEditedImage(generatedEditResult);
      setAppState('DONE');
    } catch (e) {
      setError('Failed to generate the final image. Please try again.');
      setAppState('MASK_REVIEW');
      console.error(e);
    }
  }, [originalImage, editPrompt, inferenceSteps, seed]);

  const handleDownload = () => {
    if (editedImage) {
      const link = document.createElement('a');
      link.href = editedImage;
      link.download = 'edited-image.png';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  const handleReset = () => {
    setOriginalImage(null);
    setMaskImage(null);
    setEditedImage(null);
    setError(null);
    setSegmentationPrompt('');
    setEditPrompt('');
    setAppState('IDLE');
  };
  
  const handleHistoryChange = useCallback((state: { canUndo: boolean, canRedo: boolean }) => {
    setUndoRedoState(state);
  }, []);

  const renderMainContent = () => {
    switch (appState) {
      case 'IDLE':
        return <ImageUploader onImageUpload={handleImageUpload} />;
      case 'SEGMENTING':
        return <Loader text="ANALYSING AND SEGMENTING" />;
      case 'GENERATING':
        return <DiffusionViewer finalImage={editedImage} />;
      case 'SEGMENT_PENDING':
      case 'MASK_REVIEW':
      case 'DONE':
        if (originalImage) {
          return (
            <EditorCanvas 
              ref={editorCanvasRef}
              image={originalImage} 
              mask={maskImage} 
              brushState={brushState}
              isEditable={appState === 'MASK_REVIEW'}
              finalImage={appState === 'DONE' ? editedImage : null}
              onHistoryChange={handleHistoryChange}
            />
          );
        }
        return <ImageUploader onImageUpload={handleImageUpload} />;
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-[#E5E5E5] text-black flex flex-col md:flex-row">
      <ControlPanel
        appState={appState}
        onSegment={handleSegment}
        onGenerate={handleGenerate}
        onDownload={handleDownload}
        onReset={handleReset}
        segmentationPrompt={segmentationPrompt}
        setSegmentationPrompt={setSegmentationPrompt}
        editPrompt={editPrompt}
        setEditPrompt={setEditPrompt}
        brushState={brushState}
        setBrushState={setBrushState}
        inferenceSteps={inferenceSteps}
        setInferenceSteps={setInferenceSteps}
        seed={seed}
        setSeed={setSeed}
        undo={() => editorCanvasRef.current?.undo()}
        redo={() => editorCanvasRef.current?.redo()}
        canUndo={undoRedoState.canUndo}
        canRedo={undoRedoState.canRedo}
      />
      <main className="flex-1 flex items-center justify-center p-4 md:p-8 bg-black">
        {error && (
          <div className="absolute top-4 right-4 bg-yellow-400 text-black p-4 border-2 border-black z-50">
            <p className="font-bold">ERROR</p>
            <p>{error}</p>
          </div>
        )}
        <div className="w-full h-full max-w-[100vh] max-h-[calc(100vh-4rem)] aspect-square border-2 border-dashed border-[#E5E5E5] flex items-center justify-center">
            {renderMainContent()}
        </div>
      </main>
    </div>
  );
};

export default App;
