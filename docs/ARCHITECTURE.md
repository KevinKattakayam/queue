# Architecture

```mermaid
flowchart LR
    V[Video frames] --> D[Ultralytics person detector]
    D --> T[ByteTrack or BoT-SORT]
    T --> F[Foot point and trajectory smoothing]
    F --> R[Queue and service ROI checks]
    R --> S[Per-visit state machine]
    S --> L[CSV and JSON event logger]
    S --> A[Video annotator]
    A --> O[Annotated video]
    L --> O2[Event and transition files]
```

The video frame index divided by the source fps is the time base. Detector-specific code is isolated from the pure state machine. Event field names are centralized in the event logger.
