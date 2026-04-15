#!/bin/bash
# Setup automated backups for CognitionSim
# This script configures cron jobs for automated backups

set -e

echo "🔧 Setting up automated backups..."

# Create backups directory
mkdir -p /app/backups
chmod 755 /app/backups

# Make backup script executable
chmod +x /app/scripts/backup_models.sh

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "📦 Docker environment detected"
    
    # Install cron if not present
    if ! command -v cron &> /dev/null; then
        echo "Installing cron..."
        apt-get update && apt-get install -y cron
    fi
    
    # Create crontab entry
    cat > /tmp/quadra-crontab << 'EOF'
# CognitionSim Automated Backups

# Hourly model backups (every hour)
0 * * * * /app/scripts/backup_models.sh >> /app/logs/backup.log 2>&1

# Daily database backups (2 AM)
0 2 * * * cd /app && sqlite3 dashboard_state/atlantean_hrm.db ".backup backups/db_backup_$(date +\%Y\%m\%d).db" >> /app/logs/backup.log 2>&1

# Weekly full backup (Sunday 3 AM)
0 3 * * 0 /app/scripts/full_backup.sh >> /app/logs/backup.log 2>&1

# Cleanup old backups (daily at 4 AM) - keep last 30 days
0 4 * * * find /app/backups -name "*.tar.gz" -mtime +30 -delete >> /app/logs/backup.log 2>&1
0 4 * * * find /app/backups -name "*.db" -mtime +30 -delete >> /app/logs/backup.log 2>&1

EOF
    
    # Install crontab
    crontab /tmp/quadra-crontab
    rm /tmp/quadra-crontab
    
    # Start cron service
    service cron start
    
    echo "✅ Cron jobs installed and service started"
    
else
    echo "💻 Host system detected"
    echo "Adding crontab entries..."
    
    # Add to user's crontab
    (crontab -l 2>/dev/null || echo "") | grep -v "atlantean_hrm" | cat - << 'EOF' | crontab -
# CognitionSim Automated Backups
0 * * * * cd /workspaces/Quadra-M777/CognitionSim-A.I.-main && ./scripts/backup_models.sh >> logs/backup.log 2>&1
0 2 * * * cd /workspaces/Quadra-M777/CognitionSim-A.I.-main && sqlite3 dashboard_state/atlantean_hrm.db ".backup backups/db_backup_$(date +\%Y\%m\%d).db" >> logs/backup.log 2>&1
0 3 * * 0 cd /workspaces/Quadra-M777/CognitionSim-A.I.-main && ./scripts/full_backup.sh >> logs/backup.log 2>&1
0 4 * * * find /workspaces/Quadra-M777/CognitionSim-A.I.-main/backups -name "*.tar.gz" -mtime +30 -delete
0 4 * * * find /workspaces/Quadra-M777/CognitionSim-A.I.-main/backups -name "*.db" -mtime +30 -delete
EOF
    
    echo "✅ Crontab entries added"
fi

# Show installed cron jobs
echo ""
echo "📋 Installed cron jobs:"
crontab -l | grep -A5 "CognitionSim"

echo ""
echo "✅ Automated backup setup complete!"
echo ""
echo "Backup schedule:"
echo "  - Hourly:  Model artifacts backup"
echo "  - Daily:   Database backup (2 AM)"
echo "  - Weekly:  Full system backup (Sunday 3 AM)"
echo "  - Daily:   Cleanup old backups (4 AM, keeps 30 days)"
