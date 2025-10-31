
import React from 'react';
import type { AppState, BrushState, BrushMode } from '../types';

interface ControlPanelProps {
  appState: AppState;
  onSegment: () => void;
  onGenerate: () => void;
  onDownload: () => void;
  onReset: () => void;
  segmentationPrompt: string;
  setSegmentationPrompt: (s: string) => void;
  editPrompt: string;
  setEditPrompt: (s: string) => void;
  brushState: BrushState;
  setBrushState: (bs: BrushState) => void;
  inferenceSteps: number;
  setInferenceSteps: (n: number) => void;
  seed: number;
  setSeed: (n: number) => void;
  undo: () => void;
  redo: () => void;
  canUndo?: boolean;
  canRedo?: boolean;
}

const Section: React.FC<{ title: string; number: number; children: React.ReactNode; active: boolean; }> = ({ title, number, children, active }) => (
  <div className={`border-b-2 border-black p-6 transition-opacity duration-300 ${active ? 'opacity-100' : 'opacity-40 pointer-events-none'}`}>
    <h2 className="text-sm font-black tracking-widest mb-4">STEP {number}: {title}</h2>
    <div className="space-y-4">
      {children}
    </div>
  </div>
);

const BrutalistInput: React.FC<React.InputHTMLAttributes<HTMLInputElement>> = (props) => (
  <input
    {...props}
    className="w-full bg-transparent border-2 border-black p-3 text-sm focus:outline-none focus:bg-yellow-400 placeholder-neutral-600 disabled:opacity-50"
  />
);

const BrutalistButton: React.FC<React.ButtonHTMLAttributes<HTMLButtonElement>> = ({ children, ...props }) => (
  <button
    {...props}
    className="w-full bg-black text-white p-4 text-sm font-bold tracking-wider hover:bg-yellow-400 hover:text-black focus:outline-none focus:bg-yellow-400 focus:text-black disabled:bg-neutral-600 disabled:text-neutral-400 disabled:cursor-not-allowed transition-colors duration-200"
  >
    {children}
  </button>
);

const Slider: React.FC<{ label: string; value: number; min: number; max: number; step: number; onChange: (e: React.ChangeEvent<HTMLInputElement>) => void; disabled?: boolean }> = 
({ label, value, min, max, step, onChange, disabled }) => (
  <div>
    <div className="flex justify-between items-baseline mb-1">
      <label className="text-xs font-bold tracking-wider">{label}</label>
      <span className="text-sm font-mono">{value}</span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={onChange}
      disabled={disabled}
      className="w-full h-2 bg-transparent appearance-none cursor-pointer disabled:cursor-not-allowed [&::-webkit-slider-runnable-track]:h-1 [&::-webkit-slider-runnable-track]:bg-black [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-5 [&::-webkit-slider-thumb]:w-5 [&::-webkit-slider-thumb]:bg-black [&::-webkit-slider-thumb]:-mt-2 [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-white focus:[&::-webkit-slider-thumb]:bg-yellow-400"
    />
  </div>
);

export const ControlPanel: React.FC<ControlPanelProps> = ({
  appState, onSegment, onGenerate, onDownload, onReset,
  segmentationPrompt, setSegmentationPrompt, editPrompt, setEditPrompt,
  brushState, setBrushState, inferenceSteps, setInferenceSteps,
  seed, setSeed, undo, redo, canUndo, canRedo,
}) => {
  const isSegmentActive = appState === 'SEGMENT_PENDING';
  const isMaskReviewActive = appState === 'MASK_REVIEW';
  const isDone = appState === 'DONE';

  const setBrushMode = (mode: BrushMode) => setBrushState({ ...brushState, mode });
  const setBrushSize = (size: number) => setBrushState({ ...brushState, size });

  return (
    <aside className="w-full md:w-[380px] bg-[#E5E5E5] border-r-2 border-black flex flex-col flex-shrink-0">
      <div className="p-6 border-b-2 border-black">
        <h1 className="text-2xl font-black">BRUTALIST AI EDITOR</h1>
        <p className="text-xs tracking-wider">RAW AI IMAGE MANIPULATION</p>
      </div>

      <div className="flex-1 overflow-y-auto">
        <Section title="SEGMENTATION" number={1} active={isSegmentActive}>
          <p className="text-xs">ENTER A PROMPT TO IDENTIFY AND MASK AN OBJECT IN YOUR UPLOADED IMAGE.</p>
          <BrutalistInput
            type="text"
            placeholder="E.G., 'THE RED CAR'"
            value={segmentationPrompt}
            onChange={(e) => setSegmentationPrompt(e.target.value)}
            disabled={!isSegmentActive}
          />
          <BrutalistButton onClick={onSegment} disabled={!isSegmentActive || !segmentationPrompt}>
            GENERATE MASK
          </BrutalistButton>
        </Section>

        <Section title="EDIT MASK AND GENERATE" number={2} active={isMaskReviewActive}>
          <div className="grid grid-cols-2 gap-2">
            <button onClick={() => setBrushMode('ADD')} className={`p-2 border-2 border-black text-xs font-bold ${brushState.mode === 'ADD' ? 'bg-black text-white' : 'bg-transparent text-black'}`}>ADD</button>
            <button onClick={() => setBrushMode('SUBTRACT')} className={`p-2 border-2 border-black text-xs font-bold ${brushState.mode === 'SUBTRACT' ? 'bg-black text-white' : 'bg-transparent text-black'}`}>SUBTRACT</button>
          </div>
          <Slider label="BRUSH SIZE" value={brushState.size} min={5} max={100} step={1} onChange={e => setBrushSize(parseInt(e.target.value))} disabled={!isMaskReviewActive}/>
          <div className="grid grid-cols-2 gap-2">
             <button onClick={undo} disabled={!canUndo} className="p-2 border-2 border-black text-xs font-bold disabled:opacity-40 disabled:cursor-not-allowed">UNDO</button>
             <button onClick={redo} disabled={!canRedo} className="p-2 border-2 border-black text-xs font-bold disabled:opacity-40 disabled:cursor-not-allowed">REDO</button>
          </div>
          <hr className="border-t-2 border-dashed border-black my-2" />
          <p className="text-xs">ENTER A PROMPT TO TRANSFORM THE MASKED AREA.</p>
          <BrutalistInput
            type="text"
            placeholder="E.G., 'A TIGER WITH SUNGLASSES'"
            value={editPrompt}
            onChange={(e) => setEditPrompt(e.target.value)}
            disabled={!isMaskReviewActive}
          />
          <Slider label="INFERENCE STEPS" value={inferenceSteps} min={10} max={100} step={1} onChange={e => setInferenceSteps(parseInt(e.target.value))} disabled={!isMaskReviewActive}/>
          <Slider label="SEED" value={seed} min={0} max={10000} step={1} onChange={e => setSeed(parseInt(e.target.value))} disabled={!isMaskReviewActive}/>
          <BrutalistButton onClick={onGenerate} disabled={!isMaskReviewActive || !editPrompt}>
            GENERATE IMAGE
          </BrutalistButton>
        </Section>

        <Section title="EXPORT" number={3} active={isDone}>
          <BrutalistButton onClick={onDownload} disabled={!isDone}>
            DOWNLOAD PNG
          </BrutalistButton>
          <button onClick={onReset} className="w-full p-2 text-xs font-bold underline decoration-2 decoration-yellow-400 hover:bg-yellow-400">
            START OVER
          </button>
        </Section>
      </div>
    </aside>
  );
};
