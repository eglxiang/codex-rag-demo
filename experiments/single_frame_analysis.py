"""Independent single-frame experiment using the shared lazy backend."""
if __package__:
    from .visual_backend import LocalVisualBackend, visual_prompt
else:
    from visual_backend import LocalVisualBackend, visual_prompt

if __name__ == "__main__":
    print(LocalVisualBackend().generate(
        ["outputs/frames/frames/frame_00000750.jpg"], visual_prompt()))
