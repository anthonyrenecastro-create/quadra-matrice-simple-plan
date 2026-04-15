#!/bin/bash
# Full backup script for CognitionSim
# Creates a complete backup of all system components

set -e

BACKUP_DIR="/app/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="full_backup_${TIMESTAMP}"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"

echo "🔄 Starting full system backup..."
echo "Timestamp: ${TIMESTAMP}"

# Create temporary directory
TEMP_DIR="/tmp/${BACKUP_NAME}"
mkdir -p "${TEMP_DIR}"

# Backup models
echo "📦 Backing up models..."
if [ -d "/app/models" ]; then
    cp -r /app/models "${TEMP_DIR}/"
fi

# Backup database
echo "💾 Backing up database..."
if [ -f "/app/dashboard_state/atlantean_hrm.db" ]; then
    mkdir -p "${TEMP_DIR}/database"
    sqlite3 /app/dashboard_state/atlantean_hrm.db ".backup '${TEMP_DIR}/database/atlantean_hrm.db'"
fi

# Backup dashboard state
echo "📊 Backing up dashboard state..."
if [ -d "/app/dashboard_state" ]; then
    cp -r /app/dashboard_state "${TEMP_DIR}/"
fi

# Backup configuration
echo "⚙️  Backing up configuration..."
if [ -f "/app/.env" ]; then
    cp /app/.env "${TEMP_DIR}/.env.backup"
fi

# Backup logs (last 7 days)
echo "📝 Backing up recent logs..."
if [ -d "/app/logs" ]; then
    mkdir -p "${TEMP_DIR}/logs"
    find /app/logs -name "*.log" -mtime -7 -exec cp {} "${TEMP_DIR}/logs/" \;
fi

# Create backup metadata
cat > "${TEMP_DIR}/backup_metadata.json" << EOF
{
  "backup_timestamp": "${TIMESTAMP}",
  "backup_type": "full",
  "hostname": "$(hostname)",
  "components": [
    "models",
    "database",
    "dashboard_state",
    "configuration",
    "logs"
  ]
}
EOF

# Create compressed archive
echo "🗜️  Compressing backup..."
cd "${TEMP_DIR}/.."
tar -czf "${BACKUP_PATH}" "${BACKUP_NAME}"

# Cleanup temp directory
rm -rf "${TEMP_DIR}"

# Verify backup
if [ -f "${BACKUP_PATH}" ]; then
    SIZE=$(du -h "${BACKUP_PATH}" | cut -f1)
    echo "✅ Full backup completed successfully"
    echo "   File: ${BACKUP_PATH}"
    echo "   Size: ${SIZE}"
    
    # Calculate checksum
    SHA256=$(sha256sum "${BACKUP_PATH}" | cut -d' ' -f1)
    echo "   SHA256: ${SHA256}"
    echo "${SHA256}  ${BACKUP_PATH}" > "${BACKUP_PATH}.sha256"
else
    echo "❌ Backup failed - file not created"
    exit 1
fi

# Cleanup old full backups (keep last 4 weekly)
echo "🧹 Cleaning up old full backups..."
cd "${BACKUP_DIR}"
ls -t full_backup_*.tar.gz 2>/dev/null | tail -n +5 | xargs -r rm -f
ls -t full_backup_*.tar.gz.sha256 2>/dev/null | tail -n +5 | xargs -r rm -f

echo "✅ Full backup process completed"
