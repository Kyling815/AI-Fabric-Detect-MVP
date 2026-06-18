# Frontend Architecture: AI Fabric Detect

## Overview
The frontend is a React-based single-page application built with Vite and TypeScript. It is designed to interface with the AI Fabric Detect backend to provide a real-time inspection, reference management, and feedback loop for fabric quality control.

## Key Features
- **Real-time Inspection:** Users can drop fabric images into the inspection zone to analyze them using the AI backend.
- **Visual Feedback:** Bounding boxes are drawn directly onto the inspected images to highlight detected defects, along with confidence scores.
- **Reference Library:** Users can manage and select different reference fabric sets for more accurate anomaly detection.
- **System Monitoring:** A real-time status indicator checks backend health every 10 seconds.
- **Logging:** A live detection pipeline log displays ongoing system operations, errors, and warnings.

## Structure
- `src/`: Core source code.
  - `components/`: Reusable UI components.
    - `InspectionBay.tsx`: Handles image upload, drag-and-drop, and rendering the inspection results with SVG overlays.
    - `ReferenceList.tsx`: Manages the selection of reference datasets.
    - `ResultLog.tsx`: Displays real-time logs from the inspection pipeline.
    - `StatusIndicator.tsx`: Monitors and displays the backend health status.
  - `services/`: API communication layer.
    - `api.ts`: Centralized service for all REST API calls to the backend (`/api` proxy), including typing definitions for responses.
  - `App.tsx`: Main application container managing layout, state, and coordination between components.
  - `App.css`, `index.css`: Styling based on an industrial dark-mode palette.
