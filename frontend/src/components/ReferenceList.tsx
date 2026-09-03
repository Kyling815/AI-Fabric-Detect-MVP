import { useEffect, useState } from 'react';
import { apiService } from '../services/api';
import type { Fabric } from '../services/api';

interface FabricSelectorProps {
  onSelect: (id: string) => void;
  selectedId: string | null;
}

export const FabricSelector = ({ onSelect, selectedId }: FabricSelectorProps) => {
  const [fabrics, setFabrics] = useState<Fabric[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  const fetchFabrics = async () => {
    setLoading(true);
    try {
      const data = await apiService.getFabrics();
      setFabrics(data.fabrics || []);
    } catch (err) {
      console.error('Failed to fetch references:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let active = true;
    apiService.getFabrics()
      .then(data => {
        if (active) setFabrics(data.fabrics || []);
      })
      .catch(err => console.error('Failed to fetch references:', err))
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="reference-list">
      <div className="panel-header">
        <span>SELECT_FABRIC</span>
        <button className="btn" style={{ padding: '2px 6px', fontSize: '10px' }} onClick={() => void fetchFabrics()}>
          REFRESH
        </button>
      </div>
      <div className="section-content">
        {loading ? (
          <div className="mono" style={{ fontSize: '11px', color: 'var(--text-dim)' }}>LOADING_DATABASE...</div>
        ) : fabrics.length === 0 ? (
          <div className="mono" style={{ fontSize: '11px', color: 'var(--accent-orange)' }}>NO_FABRICS_REGISTERED</div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {fabrics.map((fabric: Fabric, index: number) => (
              <div 
              key={fabric.fabric_id || index}
              onClick={() => fabric.inspectable && onSelect(fabric.fabric_id)}
              style={{
                padding: '8px',
                backgroundColor: selectedId === fabric.fabric_id ? 'rgba(0, 245, 255, 0.1)' : 'var(--bg-element)',
                border: `1px solid ${selectedId === fabric.fabric_id ? 'var(--accent-cyan)' : 'var(--border-dim)'}`,
                cursor: fabric.inspectable ? 'pointer' : 'not-allowed',
                opacity: fabric.inspectable ? 1 : 0.55,
                fontSize: '12px',
                fontFamily: 'var(--font-mono)',
                color: selectedId === fabric.fabric_id ? 'var(--accent-cyan)' : 'var(--text-main)',
                transition: 'all 0.2s'
              }}
              >
              <div>{fabric.display_name}</div>
              <div style={{ fontSize: '10px', color: 'var(--text-dim)', marginTop: '4px' }}>{fabric.fabric_id}</div>
              <div style={{ fontSize: '10px', color: 'var(--text-dim)', marginTop: '4px' }}>
                REF: {fabric.active_reference || 'NONE'} | STATUS: {fabric.status}
              </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
