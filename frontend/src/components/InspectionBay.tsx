import { useState, useRef } from 'react';
import type { DragEvent, ChangeEvent } from 'react';
import { apiService } from '../services/api';
import type { InspectionResponse, DefectBox } from '../services/api';

interface InspectionBayProps {
  selectedFabricId: string | null;
  onResult?: (result: InspectionResponse) => void;
}

export const InspectionBay = ({ selectedFabricId, onResult }: InspectionBayProps) => {
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [results, setResults] = useState<InspectionResponse | null>(null);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    if (!file || !file.type.startsWith('image/')) return;
    if (!selectedFabricId) {
      alert('SELECT_FABRIC_BEFORE_SCAN');
      return;
    }
    
    setIsUploading(true);
    // Preview local image
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        setImageSrc(reader.result);
      }
    };
    reader.readAsDataURL(file);

    try {
      const result = await apiService.inspectImage(file, selectedFabricId);
      setResults(result);
      if (onResult) onResult(result);
    } catch (err: unknown) {
      console.error('Inspection failed:', err);
      const message = err instanceof Error ? err.message : 'Unknown error';
      alert('INSPECTION_FAILURE: ' + message);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDrag = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="inspection-bay" style={{ flex: 1, display: 'flex', flexDirection: 'column', position: 'relative' }}>
      <div className="panel-header" style={{ borderBottom: '1px solid var(--border-dim)' }}>
        <span>ACTIVE_INSPECTION_ZONE</span>
        {isUploading && <span className="mono" style={{ color: 'var(--accent-cyan)', animation: 'pulse 1s infinite' }}>[PROCESSING...]</span>}
      </div>

      <div 
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '40px',
          position: 'relative',
          backgroundColor: dragActive ? 'rgba(0, 245, 255, 0.02)' : 'transparent',
          transition: 'all 0.2s'
        }}
      >
        {/* Decorative Grid Overlays */}
        <div style={{ position: 'absolute', top: 20, left: 20, width: 40, height: 40, borderTop: '2px solid var(--border-bright)', borderLeft: '2px solid var(--border-bright)' }} />
        <div style={{ position: 'absolute', top: 20, right: 20, width: 40, height: 40, borderTop: '2px solid var(--border-bright)', borderRight: '2px solid var(--border-bright)' }} />
        <div style={{ position: 'absolute', bottom: 20, left: 20, width: 40, height: 40, borderBottom: '2px solid var(--border-bright)', borderLeft: '2px solid var(--border-bright)' }} />
        <div style={{ position: 'absolute', bottom: 20, right: 20, width: 40, height: 40, borderBottom: '2px solid var(--border-bright)', borderRight: '2px solid var(--border-bright)' }} />

        {!imageSrc ? (
          <div style={{ textAlign: 'center' }}>
            <div className="mono" style={{ fontSize: '14px', color: 'var(--text-dim)', marginBottom: '20px' }}>
              DROP_FABRIC_SAMPLE_FOR_ANALYSIS
            </div>
            <button className="btn primary" disabled={!selectedFabricId} onClick={() => fileInputRef.current?.click()}>
              {selectedFabricId ? 'INITIALIZE_SCAN' : 'SELECT_FABRIC_FIRST'}
            </button>
          </div>
        ) : (
          <div style={{ position: 'relative', maxWidth: '100%', maxHeight: '100%' }}>
            <img 
              src={imageSrc} 
              alt="Fabric Scan" 
              style={{ display: 'block', maxWidth: '100%', maxHeight: '70vh', border: '1px solid var(--border-dim)' }} 
            />
            
            {/* SVG Overlay for Bounding Boxes */}
            {results?.image_metadata && results.boxes.length > 0 && (
              <svg 
                style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
                viewBox={`0 0 ${results.image_metadata.width} ${results.image_metadata.height}`}
                preserveAspectRatio="xMidYMid meet"
              >
                {results.boxes.map((defect: DefectBox, i: number) => (
                  <g key={i}>
                    <rect 
                      x={defect.x}
                      y={defect.y}
                      width={defect.w}
                      height={defect.h}
                      fill="none" 
                      stroke="var(--accent-orange)" 
                      strokeWidth="2"
                    />
                    <text 
                      x={defect.x}
                      y={Math.max(12, defect.y - 5)}
                      fill="var(--accent-orange)" 
                      fontSize="12" 
                      fontFamily="var(--font-mono)"
                    >
                      DEFECT_{i} [SCORE {defect.score.toFixed(3)}]
                    </text>
                  </g>
                ))}
              </svg>
            )}
            
            <button 
              className="btn danger" 
              style={{ position: 'absolute', top: '-40px', right: 0 }}
              onClick={() => { setImageSrc(null); setResults(null); }}
            >
              CLEAR_BUFFER
            </button>
          </div>
        )}
      </div>

      <input 
        type="file" 
        ref={fileInputRef} 
        style={{ display: 'none' }} 
        onChange={(e: ChangeEvent<HTMLInputElement>) => {
            if (e.target.files && e.target.files[0]) {
                handleFile(e.target.files[0]);
            }
        }}
        accept="image/*"
      />
    </div>
  );
};
