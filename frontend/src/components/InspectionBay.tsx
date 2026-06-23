import { useState, useRef } from 'react';
import type { DragEvent, ChangeEvent } from 'react';
import { apiService } from '../services/api';
import type { InspectionResponse, Defect } from '../services/api';

interface InspectionBayProps {
  selectedReferenceId: string | null;
  onResult?: (result: InspectionResponse) => void;
}

export const InspectionBay = ({ selectedReferenceId, onResult }: InspectionBayProps) => {
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [results, setResults] = useState<InspectionResponse | null>(null);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    console.log('handleFile called with file:', file);
    if (!file || !file.type.startsWith('image/')) return;
    
    setIsUploading(true);
    // Preview local image
    const reader = new FileReader();
    reader.onload = () => {
      console.log('FileReader onload called');
      if (typeof reader.result === 'string') {
        console.log('Setting imageSrc');
        setImageSrc(reader.result);
      } else {
        console.log('FileReader result is not a string:', reader.result);
      }
    };
    reader.readAsDataURL(file);

    try {
      console.log('Calling apiService.inspectImage');
      const result = await apiService.inspectImage(file, selectedReferenceId);
      console.log('API inspection result:', result);
      setResults(result);
      if (onResult) onResult(result);
    } catch (err: any) {
      console.error('Inspection failed:', err);
      alert('INSPECTION_FAILURE: ' + (err.message || 'Unknown error'));
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
            <button className="btn primary" onClick={() => fileInputRef.current?.click()}>
              INITIALIZE_SCAN
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
            {results && results.defects && Array.isArray(results.defects) && (
              <svg 
                style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
                viewBox={`0 0 ${results.width || 100} ${results.height || 100}`}
                preserveAspectRatio="xMidYMid meet"
              >
                {results.defects.map((defect: Defect, i: number) => (
                  <g key={i}>
                    <rect 
                      x={defect.bbox[0]} 
                      y={defect.bbox[1]} 
                      width={defect.bbox[2] - defect.bbox[0]} 
                      height={defect.bbox[3] - defect.bbox[1]} 
                      fill="none" 
                      stroke="var(--accent-orange)" 
                      strokeWidth="2"
                    />
                    <text 
                      x={defect.bbox[0]} 
                      y={defect.bbox[1] - 5} 
                      fill="var(--accent-orange)" 
                      fontSize="12" 
                      fontFamily="var(--font-mono)"
                    >
                      DEFECT_{i} [{Math.round(defect.confidence * 100)}%]
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
