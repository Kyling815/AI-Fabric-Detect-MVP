import { useEffect, useState } from 'react';
import { apiService } from '../services/api';

export const StatusIndicator = () => {
  const [status, setStatus] = useState('OFFLINE');
  const [lastCheck, setLastCheck] = useState<string | null>(null);

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const data = await apiService.checkHealth();
        if (data.status === 'ok') {
          setStatus('ONLINE');
        } else {
          setStatus('ERROR');
        }
      } catch {
        setStatus('OFFLINE');
      }
      setLastCheck(new Date().toLocaleTimeString());
    };

    checkStatus();
    const interval = setInterval(checkStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = () => {
    switch (status) {
      case 'ONLINE': return 'var(--accent-green)';
      case 'ERROR': return 'var(--accent-orange)';
      case 'OFFLINE': return 'var(--accent-red)';
      default: return 'var(--text-dim)';
    }
  };

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
      <div style={{ 
        width: '8px', 
        height: '8px', 
        borderRadius: '50%', 
        backgroundColor: getStatusColor(),
        boxShadow: `0 0 8px ${getStatusColor()}`,
        transition: 'all 0.3s'
      }} />
      <span style={{ color: getStatusColor(), fontWeight: 'bold' }}>SYSTEM_{status}</span>
      <span style={{ color: 'var(--text-dim)' }}>[{lastCheck || '--:--:--'}]</span>
    </div>
  );
};
