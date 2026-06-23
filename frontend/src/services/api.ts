export interface Defect {
  bbox: [number, number, number, number];
  confidence: number;
}

export interface InspectionResponse {
  inspection_id: string;
  defects: Defect[];
  width: number;
  height: number;
}

export interface Reference {
  reference_id: string;
  processed_image_count: number;
  status?: string;
}

export interface ModelInfo {
  model_name: string;
  architecture: string;
  input_size: string;
}

const BASE_URL = '/api';

const handleResponse = async <T,>(response: Response): Promise<T> => {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `Request failed with status ${response.status}`);
  }
  return response.json();
};

export const apiService = {
  // 1. Health Check
  checkHealth: (): Promise<{status: string}> => 
    fetch(`${BASE_URL}/health`).then(res => handleResponse<{status: string}>(res)),

  // 2. Inspections (The AI Core)
  inspectImage: (imageFile: File, referenceId: string | null = null): Promise<InspectionResponse> => {
    const formData = new FormData();
    formData.append('image', imageFile); 
    if (referenceId) formData.append('reference_id', referenceId);

    return fetch(`${BASE_URL}/inspect`, {
      method: 'POST',
      body: formData,
    }).then(res => handleResponse<InspectionResponse>(res));
  },

  getInspections: (): Promise<{inspections: string[]}> => 
    fetch(`${BASE_URL}/inspections`).then(res => handleResponse<{inspections: string[]}>(res)),

  getInspectionById: (id: string): Promise<InspectionResponse> => 
    fetch(`${BASE_URL}/inspections/${id}`).then(res => handleResponse<InspectionResponse>(res)),

  // 3. Reference Library
  buildReference: (requestData: {dataset_id: string}): Promise<{status: string}> => 
    fetch(`${BASE_URL}/reference/build`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestData),
    }).then(res => handleResponse<{status: string}>(res)),

  getReferenceStatus: (id: string | null = null): Promise<{status: string}> => {
    const url = id ? `${BASE_URL}/reference/status/${id}` : `${BASE_URL}/reference/status`;
    return fetch(url).then(res => handleResponse<{status: string}>(res));
  },

  getReferenceList: (): Promise<{references: Reference[]}> => 
    fetch(`${BASE_URL}/reference/list`).then(res => handleResponse<{references: Reference[]}>(res)),

  // 4. Datasets
  getDatasetPaths: (): Promise<{paths: string[]}> => 
    fetch(`${BASE_URL}/datasets/paths`).then(res => handleResponse<{paths: string[]}>(res)),

  discoverDatasets: (): Promise<{datasets: string[]}> => 
    fetch(`${BASE_URL}/datasets/discover`).then(res => handleResponse<{datasets: string[]}>(res)),

  // 5. Feedback
  submitFeedback: (feedbackData: {id: string, note: string}): Promise<{status: string}> => 
    fetch(`${BASE_URL}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(feedbackData),
    }).then(res => handleResponse<{status: string}>(res)),

  // 6. Model info
  getModelInfo: (): Promise<ModelInfo> => 
    fetch(`${BASE_URL}/model-info`).then(res => handleResponse<ModelInfo>(res)),
};
