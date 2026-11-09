# Records docs/demo/report.gif: `vhs docs/demo/report.tape` from the repo root.
# Renders the committed results (no model needed).
Output docs/demo/report.gif
Set WaitTimeout 600s
Set WaitPattern /\$\s*$/
Set Shell "bash"
Set FontSize 13
Set Width 1300
Set Height 400
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && source .venv/bin/activate && clear` Enter
Sleep 2s
Show
Type `python -m prompt_optimizer report | sed -n '/^## gsm8k/,/^Test examples/p'` Enter
Wait
Sleep 2s
