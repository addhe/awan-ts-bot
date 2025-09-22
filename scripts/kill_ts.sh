#!/bin/bash

# Get the process ID of the run_ts.sh script
PID_SCRIPT=$(pgrep -f "scripts/run_ts.sh")

# Get the process ID of main.py
PID=$(pgrep -f "app/main.py")

# Check if the process is running and terminate it
if [ -z "$PID_SCRIPT" ]; then
  echo "Script runner (run_ts.sh) is not running."
else
  echo "Killing script runner with PID: $PID_SCRIPT"
  kill $PID_SCRIPT
fi

if [ -z "$PID" ]; then
  echo "Trading bot (main.py) is not running."
else
  echo "Killing trading bot with PID: $PID"
  kill $PID
fi
