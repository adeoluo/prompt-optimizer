# Records docs/demo/evaluate.gif from a real run: `vhs docs/demo/evaluate.tape` from the repo root.
Output docs/demo/evaluate.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 420
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate && clear` Enter
Sleep 10s
Show
Type `python -m prompt_optimizer evaluate` Enter
Sleep 3s
