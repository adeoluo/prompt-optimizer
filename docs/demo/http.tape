# Records docs/demo/http.gif from a real run: `vhs docs/demo/http.tape` from the repo root.
Output docs/demo/http.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 440
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate` Enter
Sleep 2s
Type `python -m prompt_optimizer serve --port 18322 >/dev/null 2>&1 &` Enter
Sleep 2s
Sleep 8.0s
Type `clear` Enter
Sleep 1s
Show
Type `curl -s localhost:18322/health; echo` Enter
Sleep 1.5s
Type `curl -s localhost:18322/predict -d '{"text": "I need a refund for my money"}' | jq -c` Enter
Sleep 3s
Type `curl -s localhost:18322/predict -d '{"text": "Please cancel my subscription"}' | jq -c` Enter
Sleep 3s
Hide
Type `kill %1` Enter
Sleep 1s
