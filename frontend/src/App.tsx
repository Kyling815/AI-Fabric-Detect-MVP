import { useState, useEffect } from 'react';
import './App.css';
import { apiService } from './services/api';
import { StatusIndicator } from './components/StatusIndicator';
import { ReferenceList } from './components/ReferenceList';
import { InspectionBay } from './components/InspectionBay';
import { ResultLog } from './components/ResultLog';

function App() {
  const [selectedReferenceId, setSelectedReferenceId] = useState(null);
  const [logs, setLogs] = useState<any[]>([]);
  const [modelInfo, setModelInfo] = useState<any>(null);

  useEffect(() => {
    // Initial load
    addLog('SYSTEM_INITIALIZING', 'INFO');
    apiService.getModelInfo()
      .then(info => {
        setModelInfo(info);
        addLog(`MODEL_LOADED: ${info.model_name || 'UNKNOWN'}`, 'INFO');
      })
      .catch(err => addLog(`MODEL_ERROR: ${err.message}`, 'error'));
  }, []);

  const addLog = (message: string, level = 'INFO') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev, { timestamp, message, level, type: level.toLowerCase() }]);
  };

  const handleInspectionResult = (result: any) => {
    addLog('SCAN_COMPLETE', 'INFO');
    if (result.defects && result.defects.length > 0) {
      addLog(`DEFECTS_DETECTED: ${result.defects.length} clusters found`, 'warning');
    } else {
      addLog('SCAN_RESULT: NO_DEFECTS_FOUND', 'INFO');
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <h1 style={{ fontSize: '18px', margin: 0 }}>AI_FABRIC_DETECT // V0.1</h1>
          <StatusIndicator />
        </div>
        <div className="mono" style={{ fontSize: '12px', color: 'var(--text-dim)' }}>
          ENV: {import.meta.env.MODE.toUpperCase()} | TARGET: LOCAL_SERVER
        </div>
      </header>

      <aside className="sidebar-left">
        <ReferenceList 
          selectedId={selectedReferenceId} 
          onSelect={(id: any) => {
            setSelectedReferenceId(id);
            addLog(`REFERENCE_SWITCHED: ${id}`, 'INFO');
          }} 
        />
        
        <div style={{ flex: 1 }} />
        
        <div className="panel-header">SYSTEM_CONTROLS</div>
        <div className="section-content" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <button className="btn" onClick={() => addLog('SYSTEM_SELF_TEST_START', 'INFO')}>RUN_DIAGNOSTICS</button>
          <button className="btn danger" onClick={() => setLogs([])}>PURGE_LOGS</button>
        </div>
      </aside>

      <main className="main-content">
        <InspectionBay 
          selectedReferenceId={selectedReferenceId} 
          onResult={handleInspectionResult} 
        />
      </main>

      <aside className="sidebar-right">
        <div className="panel-header">MODEL_SPECIFICATIONS</div>
        <div className="section-content mono" style={{ fontSize: '12px' }}>
          {modelInfo ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div><span style={{ color: 'var(--text-dim)' }}>NAME:</span> {modelInfo.model_name}</div>
              <div><span style={{ color: 'var(--text-dim)' }}>ARCH:</span> {modelInfo.architecture || 'RESNET'}</div>
              <div><span style={{ color: 'var(--text-dim)' }}>INPUT:</span> {modelInfo.input_size || '224x224'}</div>
              <div><span style={{ color: 'var(--text-dim)' }}>PRECISION:</span> FP32</div>
            </div>
          ) : (
            <div style={{ color: 'var(--text-dim)' }}>QUERYING_MODEL_STATE...</div>
          )}
        </div>

        <div className="panel-header">INSPECTION_METRICS</div>
        <div className="section-content">
           <div style={{ height: '100px', border: '1px dashed var(--border-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-dim)', fontSize: '10px', textAlign: 'center' }}>
             AWAITING_SCAN_DATA_FOR_METRIC_VISUALIZATION
           </div>
        </div>

        <ResultLog logs={logs} />
      </aside>

      <footer className="footer">
        <div style={{ flex: 1 }}>CONNECTED_TO_DME_BACKEND // SOCKET_STABLE</div>
        <div>(C) 2026 AI_FABRIC_LABS // PROTOTYPE_BUILD</div>
      </footer>
    </div>
  );
}

export default App;
