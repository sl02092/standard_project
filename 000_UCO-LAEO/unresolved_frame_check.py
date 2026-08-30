from ucolaeo_temporal_utils import build_person_timelines
timelines, _ = build_person_timelines()
frames = timelines[('got01', '3')]['frames']
unresolved = sorted(f for f, v in frames.items() if v['gaze_type'] is None)
print(unresolved[:10])