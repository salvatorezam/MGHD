#!/bin/bash
# Helper script to launch the big training run inside a tmux session.

SESSION_NAME="big_run"

# Check if session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
  # Create new session in detached mode
  tmux new-session -d -s $SESSION_NAME
  
  # Send the run command to the tmux session
  # We assume the user is in the root of the workspace, so we use the relative path
  tmux send-keys -t $SESSION_NAME "source /u/home/kulp/miniconda3/bin/activate mlqec-env" C-m
  tmux send-keys -t $SESSION_NAME "./scripts/run_final_d15.sh" C-m
  
  echo "----------------------------------------------------------------"
  echo "Training started in tmux session: '$SESSION_NAME'"
  echo "The job is running in the background and will survive disconnects."
  echo "----------------------------------------------------------------"
  echo "To view the progress, run:"
  echo "  tmux attach -t $SESSION_NAME"
  echo ""
  echo "To detach again (leave it running), press: Ctrl+B, then D"
  echo "----------------------------------------------------------------"

else
  echo "Session '$SESSION_NAME' already exists."
  echo "Attach to it using: tmux attach -t $SESSION_NAME"
fi
