#!/bin/bash

# Get the process ID of the run_ts.sh script
PID=$(pgrep -f "run_ts.sh")

# Check if the process is running and terminate it
if [ -z "$PID" ]; then
  echo "Trading bot is not running."
else
  echo "Killing trading bot with PID: $PID"
  kill $PID
fi