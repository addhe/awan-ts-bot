#!/bin/bash

# Paths to the Python script and configuration file
SCRIPT_PATH="/root/awan-ts-bot/main.py"
CONFIG_PATH="/root/awan-ts-bot/config/config.py"

# Define logging file
LOG_FILE="trade_log_spot.log"

# Function to run the bot with logging
run_bot() {
  echo "$(date): Starting trading bot" >> $LOG_FILE
  python3 "$SCRIPT_PATH"
  EXIT_CODE=$?
  echo "$(date): Trading bot stopped with exit code $EXIT_CODE" >> $LOG_FILE
}

# Initial checksums for the script and configuration file
LAST_SCRIPT_CHECKSUM=$(md5sum "$SCRIPT_PATH" | awk '{ print $1 }')
LAST_CONFIG_CHECKSUM=$(md5sum "$CONFIG_PATH" | awk '{ print $1 }')

# Infinite loop to keep the bot running and watch for changes
while true; do
  CURRENT_SCRIPT_CHECKSUM=$(md5sum "$SCRIPT_PATH" | awk '{ print $1 }')
  CURRENT_CONFIG_CHECKSUM=$(md5sum "$CONFIG_PATH" | awk '{ print $1 }')

  if [[ "$CURRENT_SCRIPT_CHECKSUM" != "$LAST_SCRIPT_CHECKSUM" ]] || [[ "$CURRENT_CONFIG_CHECKSUM" != "$LAST_CONFIG_CHECKSUM" ]]; then
    echo "$(date): Detected changes in script or configuration, reloading..." >> $LOG_FILE
    LAST_SCRIPT_CHECKSUM=$CURRENT_SCRIPT_CHECKSUM
    LAST_CONFIG_CHECKSUM=$CURRENT_CONFIG_CHECKSUM
  else
    run_bot
  fi

  # Add a sleep to wait for a few seconds before restart; this avoids immediate restart on failure.
  echo "$(date): Restarting trading bot after failure" >> $LOG_FILE
  sleep 10
done