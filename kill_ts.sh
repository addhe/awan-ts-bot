#!/bin/bash

# Get the process ID of the run_ts.sh script
PID_SCRIPT=$(pgrep -f "run_ts.sh")

# Absolute path to directory where run_ts.sh resides
RUN_SCRIPT_DIR="/root/awan-ts-bot"

# Get the process ID of main.py started from the specific directory
PID=$(pgrep -f "${RUN_SCRIPT_DIR}/main.py")

# Check if the process is running and terminate it
if [ -z "$PID" ]; then
  echo "Trading bot is not running."
else
  echo "Killing script runner with PID: $PID_SCRIPT"
  kill $PID_SCRIPT
  echo "Killing trading bot with PID: $PID"
  kill $PID
fi