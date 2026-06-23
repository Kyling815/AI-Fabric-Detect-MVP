import { useEffect, useState } from 'react';
import { apiService } from '../services/api';
import type { Reference } from '../services/api';

interface ReferenceListProps {
  onSelect: (id: string) => void;
  selectedId: string | null;
}

export const ReferenceList = ({ onSelect, selectedId }: ReferenceListProps) => {
  const [references, setReferences] = useState<Reference[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchReferences = async () => {
      try {
        const data = await apiService.getReferenceList();
        setReferences(data.references || []);
      } catch (err) {
        console.error('Failed to fetch references:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchReferences();
  }, []);

  return (
    <div className="reference-list">
      <div className="panel-header">
        <span>REFERENCE_LIBRARY</span>
        <button className="btn" style={{ padding: '2px 6px', fontSize: '10px' }}>REFRESH</button>
      </div>
      <div className="section-content">
        {loading ? (
          <div className="mono" style={{ fontSize: '11px', color: 'var(--text-dim)' }}>LOADING_DATABASE...</div>
        ) : references.length === 0 ? (
          <div className="mono" style={{ fontSize: '11px', color: 'var(--accent-orange)' }}>NO_REFERENCES_FOUND</div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {references.map((ref: Reference, index: number) => (
              <div 
              key={ref.reference_id || index}
              onClick={() => onSelect(ref.reference_id)}
              style={{
                padding: '8px',
                backgroundColor: selectedId === ref.reference_id ? 'rgba(0, 245, 255, 0.1)' : 'var(--bg-element)',
                border: `1px solid ${selectedId === ref.reference_id ? 'var(--accent-cyan)' : 'var(--border-dim)'}`,
                cursor: 'pointer',
                fontSize: '12px',
                fontFamily: 'var(--font-mono)',
                color: selectedId === ref.reference_id ? 'var(--accent-cyan)' : 'var(--text-main)',
                transition: 'all 0.2s'
              }}
              >
              <div>{ref.reference_id}</div>
              <div style={{ fontSize: '10px', color: 'var(--text-dim)', marginTop: '4px' }}>
                SAMPLES: {ref.processed_image_count} | STATUS: {ref.status || 'built'}
              </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
