# Family muse-glimmer: porting

## ATEM/channel formats (non-ChatML)

The Glimmer port (`glimmer-example/chat/atem.loom`) is a full example of a
channel-based format. Its prompt looks like `<|start|>role<|message|>…<|eot|>`;
reasoning is addressed `to=self`, and the answer `to=user`. Its control program
parses recipient headers in `output_next`. Its kit's `chat_probe.raw_from_chat()`
rebuilds the raw output text from the channels the same way: define one in your
kit for such formats (`export LOOM_KIT=W/kit` makes the benchmark scripts use it).
