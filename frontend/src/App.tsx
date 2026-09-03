import { useCallback, useEffect, useState } from 'react';
import './App.css';
import { apiService } from './services/api';
import type { InspectionResponse, ModelInfo } from './services/api';
import { StatusIndicator } from './components/StatusIndicator';
import { FabricSelector } from './components/ReferenceList';
import { InspectionBay } from './components/InspectionBay';
import { ResultLog } from './components/ResultLog';
import type { LogEntry } from './components/ResultLog';

function App() {
  const [selectedFabricId, setSelectedFabricId] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>(() => [{
    timestamp: new Date().toLocaleTimeString(),
    message: 'SYSTEM_INITIALIZING',
    level: 'INFO',
    type: 'info',
  }]);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [lastInspection, setLastInspection] = useState<InspectionResponse | null>(null);

  const addLog = useCallback((message: string, level: LogEntry['level'] = 'INFO') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev, { timestamp, message, level, type: level.toLowerCase() as LogEntry['type'] }]);
  }, []);

  useEffect(() => {
    apiService.getModelInfo()
      .then(info => {
        setModelInfo(info);
        addLog(`MODEL_LOADED: ${info.model_name || 'UNKNOWN'}`, 'INFO');
      })
      .catch(err => addLog(`MODEL_ERROR: ${err.message}`, 'error'));
  }, [addLog]);

  const handleInspectionResult = (result: InspectionResponse) => {
    setLastInspection(result);
    addLog('SCAN_COMPLETE', 'INFO');
    if (result.boxes.length > 0) {
      addLog(`DEFECTS_DETECTED: ${result.boxes.length} clusters found`, 'warning');
    } else if (result.final_result === 'NEEDS_REVIEW') {
      addLog(`SCAN_RESULT: NEEDS_REVIEW / ${result.reason || 'ANOMALY_WITHOUT_LOCALIZATION'}`, 'warning');
    } else {
      addLog('SCAN_RESULT: NO_DEFECTS_FOUND', 'INFO');
    }
  };

  const runDiagnostics = async () => {
    addLog('SYSTEM_SELF_TEST_START', 'INFO');
    try {
      const [health, info, fabrics] = await Promise.all([
        apiService.checkHealth(),
        apiService.getModelInfo(),
        apiService.getFabrics(),
      ]);
      setModelInfo(info);
      addLog(`DIAGNOSTICS_OK: ${health.status.toUpperCase()} / ${fabrics.fabrics.length} FABRICS`, 'INFO');
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      addLog(`DIAGNOSTICS_FAILED: ${message}`, 'error');
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
        <FabricSelector
          selectedId={selectedFabricId}
          onSelect={(id: string) => {
            setSelectedFabricId(id);
            addLog(`FABRIC_SELECTED: ${id}`, 'INFO');
          }} 
        />
        
        <div style={{ flex: 1 }} />
        
        <div className="panel-header">SYSTEM_CONTROLS</div>
        <div className="section-content" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <button className="btn" onClick={() => void runDiagnostics()}>RUN_DIAGNOSTICS</button>
          <button className="btn danger" onClick={() => setLogs([])}>PURGE_LOGS</button>
        </div>
      </aside>

      <main className="main-content">
        <InspectionBay
          selectedFabricId={selectedFabricId}
          onResult={handleInspectionResult} 
        />
      </main>

      <aside className="sidebar-right">
        <div className="panel-header">MODEL_SPECIFICATIONS</div>
        <div className="section-content mono" style={{ fontSize: '12px' }}>
          {modelInfo ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div><span style={{ color: 'var(--text-dim)' }}>NAME:</span> {modelInfo.model_name}</div>
              <div><span style={{ color: 'var(--text-dim)' }}>BACKBONE:</span> {modelInfo.backbone_version || 'LEGACY'}</div>
              <div><span style={{ color: 'var(--text-dim)' }}>DEVICE:</span> {modelInfo.device.toUpperCase()}</div>
              <div><span style={{ color: 'var(--text-dim)' }}>INPUT:</span> {modelInfo.model_input_size}x{modelInfo.model_input_size}</div>
              <div><span style={{ color: 'var(--text-dim)' }}>PRECISION:</span> FP32</div>
            </div>
          ) : (
            <div style={{ color: 'var(--text-dim)' }}>QUERYING_MODEL_STATE...</div>
          )}
        </div>

        <div className="panel-header">INSPECTION_METRICS</div>
        <div className="section-content">
          {lastInspection ? (
            <div className="mono" style={{ display: 'grid', gap: '8px', fontSize: '11px' }}>
              <div>RESULT: {lastInspection.final_result.toUpperCase()}</div>
              <div>FABRIC: {lastInspection.fabric_name || lastInspection.fabric_id || 'N/A'}</div>
              <div>REFERENCE: {lastInspection.reference_id || 'N/A'}</div>
              <div>ANOMALY_SCORE: {lastInspection.anomaly_score === null ? 'N/A' : lastInspection.anomaly_score.toFixed(4)}</div>
              <div>DEFECT_CLUSTERS: {lastInspection.defect_count}</div>
              <div>DEFECT_AREA: {(lastInspection.defect_area_ratio * 100).toFixed(2)}%</div>
              <div>CONFIDENCE: {lastInspection.confidence === null ? 'N/A' : `${(lastInspection.confidence * 100).toFixed(1)}%`}</div>
              {lastInspection.reason && <div>REASON: {lastInspection.reason}</div>}
            </div>
          ) : (
            <div style={{ height: '100px', border: '1px dashed var(--border-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-dim)', fontSize: '10px', textAlign: 'center' }}>
              AWAITING_SCAN_DATA_FOR_METRIC_VISUALIZATION
            </div>
          )}
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
