# Records docs/demo/compile.gif from a real run: `vhs docs/demo/compile.tape` from the repo root.
Output docs/demo/compile.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 440
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate && clear` Enter
Sleep 10s
Show
Type `python -m prompt_optimizer compile` Enter
Sleep 2s
Type `python -m prompt_optimizer predict "I need a refund for my money"` Enter
Sleep 2.5s
