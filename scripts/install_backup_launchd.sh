#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DAILY_LABEL="com.olli.betbot.backup.daily"
WEEKLY_LABEL="com.olli.betbot.backup.weekly"
LAUNCH_DIR="$HOME/Library/LaunchAgents"
mkdir -p "$LAUNCH_DIR"

DAILY_PLIST="$LAUNCH_DIR/${DAILY_LABEL}.plist"
WEEKLY_PLIST="$LAUNCH_DIR/${WEEKLY_LABEL}.plist"

cat > "$DAILY_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>${DAILY_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${ROOT}/scripts/db_backup.sh</string>
    <string>daily</string>
  </array>
  <key>WorkingDirectory</key><string>${ROOT}</string>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>2</integer>
    <key>Minute</key><integer>30</integer>
  </dict>
  <key>StandardOutPath</key><string>${ROOT}/logs/backup_daily.log</string>
  <key>StandardErrorPath</key><string>${ROOT}/logs/backup_daily.err</string>
</dict>
</plist>
PLIST

cat > "$WEEKLY_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>${WEEKLY_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${ROOT}/scripts/db_backup.sh</string>
    <string>weekly</string>
  </array>
  <key>WorkingDirectory</key><string>${ROOT}</string>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Weekday</key><integer>0</integer>
    <key>Hour</key><integer>3</integer>
    <key>Minute</key><integer>0</integer>
  </dict>
  <key>StandardOutPath</key><string>${ROOT}/logs/backup_weekly.log</string>
  <key>StandardErrorPath</key><string>${ROOT}/logs/backup_weekly.err</string>
</dict>
</plist>
PLIST

mkdir -p "$ROOT/logs"

launchctl unload "$DAILY_PLIST" 2>/dev/null || true
launchctl unload "$WEEKLY_PLIST" 2>/dev/null || true
launchctl load "$DAILY_PLIST"
launchctl load "$WEEKLY_PLIST"

echo "launchd_installed daily=$DAILY_LABEL weekly=$WEEKLY_LABEL"
