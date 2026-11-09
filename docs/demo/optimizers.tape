# Records docs/demo/optimizers.gif from a real run: `vhs docs/demo/optimizers.tape` from the repo root.
# The optimizer runs against the deterministic fake model, so the recording needs no server.
Output docs/demo/optimizers.gif
Set WaitTimeout 600s
Set WaitPattern /\$\s*$/
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 460
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate && clear` Enter
Sleep 10s
Show
Type `python -m prompt_optimizer optimizers` Enter
Wait
Sleep 2s
Type `python -m prompt_optimizer optimize --fake --optimizer gepa --budget 300` Enter
Wait
Sleep 2s
