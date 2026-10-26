# Two-minute walkthrough

1. **0:00–0:20** Show both camera views and explain that their queue/service polygons are separate because the views differ.
2. **0:20–0:45** Play `queue_video1.MOV` around 26.09–27.22 seconds. Show the pending countdown and return cancellation; the same visit resumes.
3. **0:45–1:05** Explain that neither clip produced a confirmed abandonment. There is no truthful abandonment timestamp to show; do not present a cancellation as an abandonment.
4. **1:05–1:25** Open `output/queue_video1/events.csv` and show `visit_0004`, whose 34.96–57.55 second visit links previous tracker IDs 31, 70 and 71. Explain this is the buffered identity-recovery record, and that exact switch instants are not logged.
5. **1:25–1:45** Show the CPU run summary and explain that both files processed without a GPU, but took longer than their clip durations.
6. **1:45–2:00** Open the event and transition CSVs and chart. State that precision, recall, and false-positive/false-negative explanations remain pending because no manual labels have been entered in `data/ground_truth.csv`.

The timestamps above come from the two uploaded runs. The clips show classroom movement rather than a clearly marked queue, so ROI-based visit counts are provisional until manually labeled.
