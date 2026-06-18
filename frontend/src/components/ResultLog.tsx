import { useEffect, useRef } from 'react';

export interface LogEntry {
  timestamp: string;
  message: string;
  level: 'INFO' | 'warning' | 'error';
  type: 'INFO' | 'warning' | 'error';
}

export const ResultLog = ({ logs }: { logs: LogEntry[] }) => {
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <div className="result-log" style={{ flex: 1, display: 'flex', flexDirection: 'column', backgroundColor: 'var(--bg-panel)', overflow: 'hidden' }}>
      <div className="panel-header">
        <span>DETECTION_PIPELINE_LOG</span>
      </div>
      <div style={{ 
        flex: 1, 
        padding: '12px', 
        fontFamily: 'var(--font-mono)', 
        fontSize: '10px', 
        overflowY: 'auto',
        color: 'var(--text-main)',
        lineHeight: '1.4'
      }}>
        {logs.map((log, i) => (
          <div key={i} style={{ marginBottom: '4px', borderLeft: `2px solid ${log.type === 'error' ? 'var(--accent-red)' : log.type === 'warning' ? 'var(--accent-orange)' : 'var(--border-dim)'}`, paddingLeft: '8px' }}>
            <span style={{ color: 'var(--text-dim)' }}>[{log.timestamp}]</span>{' '}
            <span style={{ color: log.type === 'error' ? 'var(--accent-red)' : log.type === 'warning' ? 'var(--accent-orange)' : 'var(--accent-cyan)' }}>
              {log.level.toUpperCase()}:
            </span>{' '}
            {log.message}
          </div>
        ))}
        {logs.length === 0 && (
          <div style={{ color: 'var(--text-dim)', fontStyle: 'italic' }}>AWAITING_INPUT_STREAM...</div>
        )}
        <div ref={logEndRef} />
      </div>
    </div>
  );
};
