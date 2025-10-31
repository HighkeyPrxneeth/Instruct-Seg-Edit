
export type AppState = 'IDLE' | 'SEGMENT_PENDING' | 'SEGMENTING' | 'MASK_REVIEW' | 'GENERATING' | 'DONE';

export type BrushMode = 'ADD' | 'SUBTRACT';

export interface BrushState {
  mode: BrushMode;
  size: number;
}

export interface EditorCanvasHandle {
  exportMask: () => string | null;
  undo: () => void;
  redo: () => void;
}
