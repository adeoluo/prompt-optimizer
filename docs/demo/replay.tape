# Records docs/demo/replay.gif: `vhs docs/demo/replay.tape` from the repo root.
# Replays a finished Qwen3.5-2B optimization run on GSM8K from the response cache:
# every model call is answered from disk, so the trajectory is the recorded one.
Output docs/demo/replay.gif
Set WaitTimeout 600s
Set WaitPattern /\$\s*$/
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 520
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && source .venv/bin/activate && export CACHE=../.bench-data/prompt-optimizer/cache/lm.sqlite && clear` Enter
Sleep 2s
Show
Type `python -m prompt_optimizer optimize --task gsm8k --optimizer gepa --budget 400 --cache $CACHE | head -12` Enter
Wait
Sleep 2s
