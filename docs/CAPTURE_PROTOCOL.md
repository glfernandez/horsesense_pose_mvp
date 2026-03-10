# Jack Stable Recording Protocol

## Objective
Capture 5-10 short clips (30-60s each) that maximize pose tracking quality for the initial offline HorseSense demo.

## Checklist
- Stable phone (tripod or leaned on railing)
- Good lighting; avoid strong backlight/silhouette
- One horse per frame for initial tests
- Keep full horse visible when possible
- Minimize handler blocking legs/head
- Prefer side view

## Recommended Shots
1. Horse standing in stall (calm)
2. Horse walking with handler (side view)
3. Horse trotting (if safe/available)
4. Grooming interaction (mostly still)
5. One hard clip with mild occlusion (robustness check)

## File Handling
- Place original clips in `data/jack_raw/`
- Create trimmed/resized versions in `data/jack_processed/`
- Use descriptive names, e.g. `jack_clip_01_sidewalk.mp4`
