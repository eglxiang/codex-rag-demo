"""Context-conditioned visual experiment using the shared lazy backend."""
if __package__:
    from .visual_backend import LocalVisualBackend, visual_prompt
else:
    from visual_backend import LocalVisualBackend, visual_prompt

if __name__ == "__main__":
    context = "This frame comes from a news report about parliamentary elections in Morocco."
    print(LocalVisualBackend().generate(
        ["outputs/frames/frames/frame_00000025.jpg"], visual_prompt(context)))
