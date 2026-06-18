# Detailed Frontend Architectural Analysis

## 1. Overview
This document provides a comprehensive breakdown of the frontend's functional structure and an audit of the current API service layer.

## 2. Structural Breakdown (Functions & Components)

### `App.tsx` (Main Container)
- **Role:** Orchestrates the overall layout, manages shared state (references, logs, model info), and coordinates communication between sub-components.
- **Key Functions:**
  - `addLog(message, level)`: Appends log messages to the `ResultLog`.
  - `handleInspectionResult(result)`: Updates logs based on inspection outcomes (detects defects).
  - `useEffect(...)`: Triggers initialization (fetches model info).

### `components/InspectionBay.tsx`
- **Role:** The primary interactive zone for image upload and detection visualization.
- **Key Functions:**
  - `handleFile(file)`: Validates the file, previews it locally using `FileReader`, and triggers `apiService.inspectImage`.
  - `handleDrag(e)`, `handleDrop(e)`: Manage drag-and-drop interactions for image uploads.

### `components/ReferenceList.tsx`
- **Role:** Displays available reference libraries and handles selection.
- **Key Functions:**
  - `useEffect(...)`: Fetches the list of references on mount.
  - `onSelect(id)`: Triggered when a user clicks a reference item to set it as active.

### `components/StatusIndicator.tsx`
- **Role:** Monitors backend health.
- **Key Functions:**
  - `useEffect(...)`: Polls `apiService.checkHealth` every 10 seconds and updates UI status.

### `components/ResultLog.tsx`
- **Role:** Displays inspection pipeline history.
- **Key Functions:**
  - `useEffect(...)`: Automatically scrolls to the bottom of the log when new entries are added.

### `services/api.ts` (Service Layer)
- **Role:** Centralized repository for all backend interaction logic. Uses a generic `handleResponse` function to normalize fetch responses.
- **Data Models:**
  - `Reference`: Represents a baseline dataset. Key fields include:
    - `reference_id`: Unique identifier for the reference.
    - `processed_image_count`: Number of images processed for this reference.
    - `status`: Build status of the reference.

---

## 3. Unused API Endpoints
The following endpoints defined in `apiService` are **not currently used** by any frontend component:

| Endpoint Method | Purpose |
| :--- | :--- |
| `getInspections` | Fetch list of past inspections |
| `getInspectionById` | Fetch details of a specific inspection |
| `buildReference` | Trigger reference library building |
| `getReferenceStatus` | Poll status of a reference build |
| `getDatasetPaths` | Retrieve available dataset paths |
| `discoverDatasets` | Discover datasets in the environment |
| `submitFeedback` | Submit feedback for an inspection |

These endpoints represent potential future features or cleanup opportunities to reduce code bloat.
