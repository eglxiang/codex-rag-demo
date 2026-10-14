"""Temporal comparison experiment using the shared lazy backend."""
if __package__:
    from .visual_backend import LocalVisualBackend, temporal_prompt
else:
    from visual_backend import LocalVisualBackend, temporal_prompt

if __name__ == "__main__":
    print(LocalVisualBackend().generate(
        ["outputs/frames/frames/frame_00000750.jpg",
         "outputs/frames/frames/frame_00000775.jpg"], temporal_prompt(30, 31)))
