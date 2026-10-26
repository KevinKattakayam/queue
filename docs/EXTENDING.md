# Extension guide

1. **Add a state or rule:** update `State` and transition logic in `state_machine.py`, then add config parameters with documented defaults in `config/config.yaml` and validation in `config_loader.py`. Add transition tests before adapting the pipeline.
2. **Add an output field:** add the field to `EVENT_FIELDS` and row serialization in `event_logger.py`, then populate it in `state_machine.py`; update the JSON row and output tests.
3. **Add a video source:** adapt the capture creation in `main.py` while keeping its frame interface, fps and dimensions explicit. Keep detector logic unchanged.
4. **Add an alert type:** add a configured alert trigger to the state/event layer and its visual treatment to `annotator.py`; preserve the event visit ID for deduplication.
5. **Swap detector or tracker:** implement the small `track(frame)` interface in `detector_tracker.py` and expose its choice through YAML. Current Ultralytics tracker choices are ByteTrack and BoT-SORT.
6. **Add ROI pairs:** append a named queue/service polygon pair to `queue_rois`; selection currently edits the first pair interactively.
