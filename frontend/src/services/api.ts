export interface DefectBox {
  x: number;
  y: number;
  w: number;
  h: number;
  score: number;
  label: string;
}

export interface ImageMetadata {
  width: number;
  height: number;
  mode: string;
  content_type: string | null;
  file_size_bytes: number;
}

export interface InspectionResponse {
  inspection_id: string;
  reference_id: string | null;
  final_result: string;
  confidence: number | null;
  defect_area_ratio: number;
  defect_count: number;
  boxes: DefectBox[];
  original_image_url: string | null;
  annotated_image_url: string | null;
  image_metadata: ImageMetadata | null;
  model_version: string;
  created_at: string | null;
}

export interface InspectionListItem {
  inspection_id: string;
  filename: string;
  final_result: string;
  defect_area_ratio: number;
  defect_count: number;
  created_at: string;
}

export interface Reference {
  reference_id: string;
  reference_version: string;
  dataset_name: string;
  fabric_type: string | null;
  processed_image_count: number;
  created_at: string;
}

export interface ModelInfo {
  model_name: string;
  device: string;
  use_pretrained_weights: boolean;
  use_projection_head: boolean;
  projection_dim: number;
  image_max_size: number;
  model_input_size: number;
  patch_size: number;
  patch_stride: number;
  batch_size: number;
}

export interface ReferenceBuildRequest {
  dataset_name: string;
  fabric_type?: string;
  normal_path?: string;
  reference_id?: string;
  max_images?: number;
  preprocessing_version?: 'v1' | 'v2';
  preprocessing_config_path?: string;
}

export interface FeedbackRequest {
  inspection_id: string;
  user_label: string;
  corrected_result?: string;
  notes?: string;
}

const BASE_URL = '/api';

const handleResponse = async <T,>(response: Response): Promise<T> => {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `Request failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
};

export const apiService = {
  checkHealth: (): Promise<{ status: string }> =>
    fetch(`${BASE_URL}/health`).then(res => handleResponse<{ status: string }>(res)),

  inspectImage: (imageFile: File, referenceId: string | null = null): Promise<InspectionResponse> => {
    const formData = new FormData();
    formData.append('image', imageFile);
    if (referenceId) formData.append('reference_id', referenceId);
    return fetch(`${BASE_URL}/inspect`, { method: 'POST', body: formData })
      .then(res => handleResponse<InspectionResponse>(res));
  },

  getInspections: (): Promise<{ items: InspectionListItem[] }> =>
    fetch(`${BASE_URL}/inspections`).then(res => handleResponse<{ items: InspectionListItem[] }>(res)),

  getInspectionById: (id: string): Promise<InspectionResponse> =>
    fetch(`${BASE_URL}/inspections/${encodeURIComponent(id)}`).then(res => handleResponse<InspectionResponse>(res)),

  buildReference: (requestData: ReferenceBuildRequest): Promise<{ status: string }> =>
    fetch(`${BASE_URL}/reference/build`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestData),
    }).then(res => handleResponse<{ status: string }>(res)),

  getReferenceStatus: (id: string | null = null): Promise<{ status: string }> => {
    const url = id ? `${BASE_URL}/reference/status/${encodeURIComponent(id)}` : `${BASE_URL}/reference/status`;
    return fetch(url).then(res => handleResponse<{ status: string }>(res));
  },

  getReferenceList: (): Promise<{ references: Reference[] }> =>
    fetch(`${BASE_URL}/reference/list`).then(res => handleResponse<{ references: Reference[] }>(res)),

  getDatasetPaths: (): Promise<{ items: Array<{ name: string; path: string; exists: boolean }> }> =>
    fetch(`${BASE_URL}/datasets/paths`).then(res => handleResponse<{ items: Array<{ name: string; path: string; exists: boolean }> }>(res)),

  discoverDatasets: (): Promise<{ datasets: Array<{ name: string; display_name: string; exists: boolean }>; total_images: number }> =>
    fetch(`${BASE_URL}/datasets/discover`).then(res => handleResponse<{ datasets: Array<{ name: string; display_name: string; exists: boolean }>; total_images: number }>(res)),

  submitFeedback: (feedbackData: FeedbackRequest): Promise<{ status: string }> =>
    fetch(`${BASE_URL}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(feedbackData),
    }).then(res => handleResponse<{ status: string }>(res)),

  getModelInfo: (): Promise<ModelInfo> =>
    fetch(`${BASE_URL}/model-info`).then(res => handleResponse<ModelInfo>(res)),
};
