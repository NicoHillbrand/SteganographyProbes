---
name: tmux
description: Use tmux to run, monitor, and control long-running commands in background windows or panes. Use when an agent needs concurrent process orchestration, log capture, or resilient execution during terminal disconnects. Do not use for tmux setup or configuration changes.
allowed-tools:
  - Bash
---

# Tmux Process Orchestration

Use this skill for runtime process management in an existing tmux session.

## Preconditions

```bash
echo "$TMUX"
tmux list-windows
```

If `$TMUX` is empty, start or attach first:

```bash
tmux new -s work
# or
tmux a -t <name>
```

## Launch Pattern (Detached Window)

```bash
tmux new-window -d -n "<job>" -c "#{pane_current_path}"
tmux send-keys -t "<job>" "<command>" C-m
```

Example:

```bash
tmux new-window -d -n "dev-server" -c "#{pane_current_path}"
tmux send-keys -t "dev-server" "pnpm dev" C-m
```

## Inspect Output

```bash
tmux capture-pane -p -t "<job>"        # visible output
tmux capture-pane -p -S - -t "<job>"   # full scrollback
```

## Control Lifecycle

```bash
tmux send-keys -t "<job>" C-c
tmux kill-window -t "<job>"
```

## Current Keybinds (from ~/.config/forge/tmux/.tmux.conf)

- `Alt+h/j/k/l`: move panes (no prefix)
- `prefix + |` / `prefix + -`: split in current path
- `prefix + Tab`: last pane toggle
- `prefix + y`: sync-panes toggle
- `prefix + Enter`: copy-mode
- `prefix + h`: cheatsheet popup

## Config

- Path: `~/.config/forge/tmux/.tmux.conf` → stowed to `~/.tmux.conf`
- Agent features: `allow-passthrough on`, `set-clipboard on` (OSC 52), `extended-keys on` (Shift+Enter), hyperlinks (OSC 8), true color + undercurl (Ghostty overrides)
- Scrollback: 200k lines — use `prefix+Enter` for vi copy-mode, then `/` to search
- New splits inherit `pane_current_path`

## Notes

- Keep one logical process per window for deterministic capture/cleanup.
- Use stable window names (`dev-server`, `watch-tests`, `logs-api`).
- For tmux setup/config work, edit `~/.config/forge/tmux/.tmux.conf` and reload with `tmux source-file ~/.tmux.conf`.