# Disaster Recovery Procedures

## Overview

This document provides comprehensive disaster recovery (DR) procedures for CognitionSim Follow these procedures to recover from various failure scenarios.

## 🆘 Emergency Contacts

- **DevOps Lead**: [Add contact]
- **ML Engineer**: [Add contact]  
- **System Administrator**: [Add contact]
- **On-Call Engineer**: [Add contact]

## 📋 Recovery Time Objectives (RTO/RPO)

- **RTO (Recovery Time Objective)**: 4 hours
- **RPO (Recovery Point Objective)**: 1 hour (based on backup frequency)
- **Maximum Tolerable Downtime**: 24 hours

---

## 🔄 Backup Strategy

### Automated Backups

**1. Model Artifacts Backup**
```bash
# Hourly backup (configure in cron)
0 * * * * /app/scripts/backup_models.sh

# Manual backup
tar -czf models_backup_$(date +%Y%m%d_%H%M%S).tar.gz /app/models/
```

**2. Database Backup**
```bash
# Daily backup (SQLite)
0 2 * * * sqlite3 /app/dashboard_state/atlantean_hrm.db ".backup '/app/backups/db_backup_$(date +%Y%m%d).db'"

# Manual backup
sqlite3 /app/dashboard_state/atlantean_hrm.db ".backup '/app/backups/db_backup.db'"
```

**3. State Directory Backup**
```bash
# Daily backup
0 3 * * * tar -czf /app/backups/state_$(date +%Y%m%d).tar.gz /app/dashboard_state/

# Manual backup
tar -czf state_backup_$(date +%Y%m%d_%H%M%S).tar.gz /app/dashboard_state/
```

### Backup Verification

```bash
# Test backup integrity
tar -tzf models_backup_YYYYMMDD_HHMMSS.tar.gz > /dev/null && echo "✅ Backup OK" || echo "❌ Backup corrupted"

# Verify database backup
sqlite3 db_backup.db "PRAGMA integrity_check;" | grep "ok" && echo "✅ Database OK" || echo "❌ Database corrupted"
```

### Backup Retention Policy

- **Hourly backups**: Keep for 7 days
- **Daily backups**: Keep for 30 days
- **Weekly backups**: Keep for 6 months
- **Monthly backups**: Keep for 2 years

---

## 🚨 Disaster Scenarios & Recovery Procedures

### Scenario 1: Application Crash

**Symptoms**: Service unresponsive, health checks failing

**Recovery Steps**:

1. **Check service status**
   ```bash
   # Docker
   docker ps -a | grep cognitionsim
   docker logs cognitionsim-ai --tail 100
   
   # Kubernetes
   kubectl get pods -n cognitionsim
   kubectl logs <pod-name> -n cognitionsim --tail=100
   ```

2. **Restart service**
   ```bash
   # Docker Compose
   docker-compose restart
   
   # Kubernetes
   kubectl rollout restart deployment/cognitionsim-ai -n cognitionsim
   ```

3. **Verify recovery**
   ```bash
   curl http://localhost:5000/health
   curl http://localhost:5000/api/health/detailed
   ```

4. **Check logs for root cause**
   ```bash
   # Review application logs
   tail -f /app/logs/atlantean_hrm.log
   
   # Check for errors
   grep -i error /app/logs/atlantean_hrm.log | tail -50
   ```

**Estimated Recovery Time**: 5-10 minutes

---

### Scenario 2: Database Corruption

**Symptoms**: Database errors, data inconsistencies, application crashes on DB operations

**Recovery Steps**:

1. **Stop the application**
   ```bash
   docker-compose down
   # or
   kubectl scale deployment/cognitionsim-ai --replicas=0 -n cognitionsim
   ```

2. **Verify database corruption**
   ```bash
   sqlite3 /app/dashboard_state/atlantean_hrm.db "PRAGMA integrity_check;"
   ```

3. **Restore from backup**
   ```bash
   # Backup current (corrupted) database
   cp /app/dashboard_state/atlantean_hrm.db /app/backups/corrupted_$(date +%Y%m%d_%H%M%S).db
   
   # Restore from most recent backup
   cp /app/backups/db_backup_YYYYMMDD.db /app/dashboard_state/atlantean_hrm.db
   
   # Verify restored database
   sqlite3 /app/dashboard_state/atlantean_hrm.db "PRAGMA integrity_check;"
   ```

4. **Restart application**
   ```bash
   docker-compose up -d
   # or
   kubectl scale deployment/cognitionsim-ai --replicas=1 -n cognitionsim
   ```

5. **Verify recovery**
   ```bash
   curl http://localhost:5000/api/status
   ```

**Estimated Recovery Time**: 15-30 minutes  
**Data Loss**: Up to 1 hour (based on backup frequency)

---

### Scenario 3: Lost Model Artifacts

**Symptoms**: Model files missing, cannot load trained models

**Recovery Steps**:

1. **Identify missing models**
   ```bash
   ls -la /app/models/
   # Check model registry
   sqlite3 /app/dashboard_state/atlantean_hrm.db "SELECT * FROM model_versions;"
   ```

2. **Restore from backup**
   ```bash
   # Extract models backup
   tar -xzf /app/backups/models_backup_YYYYMMDD_HHMMSS.tar.gz -C /
   
   # Verify model integrity
   python -c "
   import torch
   model = torch.load('/app/models/oscillator_weights.pth')
   print('✅ Model loaded successfully')
   "
   ```

3. **Update model registry**
   ```bash
   curl -X GET http://localhost:5000/api/model/versions
   ```

4. **Test model loading**
   ```bash
   curl -X POST http://localhost:5000/api/model/load/<version>
   ```

**Estimated Recovery Time**: 20-40 minutes

---

### Scenario 4: Complete System Failure

**Symptoms**: Server down, hardware failure, catastrophic data loss

**Recovery Steps**:

1. **Provision new infrastructure**
   ```bash
   # Deploy to new server/cluster
   # Update DNS/load balancer if needed
   ```

2. **Restore application code**
   ```bash
   git clone https://github.com/anthonycastro-spaceace01/Quadra-M777.git
   cd Quadra-M777/CognitionSim-main
   ```

3. **Restore configuration**
   ```bash
   # Copy .env from secure backup location
   cp /secure-backup/.env .env
   
   # Verify configuration
   python -c "from config import get_config; c = get_config('production'); print('✅ Config OK')"
   ```

4. **Restore all data**
   ```bash
   # Restore models
   tar -xzf /remote-backup/models_backup_latest.tar.gz -C ./
   
   # Restore database
   cp /remote-backup/db_backup_latest.db ./dashboard_state/atlantean_hrm.db
   
   # Restore state
   tar -xzf /remote-backup/state_backup_latest.tar.gz -C ./
   ```

5. **Deploy application**
   ```bash
   # Docker
   docker-compose up -d
   
   # Kubernetes
   kubectl apply -f k8s/deployment.yaml -n cognitionsim
   ```

6. **Verify all systems**
   ```bash
   # Health check
   curl http://localhost:5000/health
   
   # Detailed health check
   curl http://localhost:5000/api/health/detailed
   
   # Check model versions
   curl http://localhost:5000/api/model/versions
   
   # Check system status
   curl http://localhost:5000/api/status
   ```

**Estimated Recovery Time**: 2-4 hours  
**Data Loss**: Up to 1 hour

---

### Scenario 5: Configuration Corruption

**Symptoms**: Application won't start, configuration errors, invalid settings

**Recovery Steps**:

1. **Stop application**
   ```bash
   docker-compose down
   ```

2. **Restore configuration**
   ```bash
   # Restore from .env.example
   cp .env.example .env
   
   # Update with production values
   nano .env
   
   # Or restore from backup
   cp /app/backups/.env.backup .env
   ```

3. **Validate configuration**
   ```bash
   python -c "
   from config import get_config
   try:
       c = get_config('production')
       c.validate()
       print('✅ Configuration valid')
   except Exception as e:
       print(f'❌ Configuration error: {e}')
   "
   ```

4. **Restart application**
   ```bash
   docker-compose up -d
   ```

**Estimated Recovery Time**: 10-20 minutes

---

## 📝 Post-Recovery Checklist

After recovering from any disaster:

- [ ] Verify all services are running
- [ ] Check health endpoints
- [ ] Verify database integrity
- [ ] Test model loading
- [ ] Review logs for errors
- [ ] Test API endpoints
- [ ] Monitor system metrics
- [ ] Document incident
- [ ] Update runbooks if needed
- [ ] Conduct post-mortem
- [ ] Update backup procedures if needed

---

## 🧪 Disaster Recovery Testing

**Schedule**: Quarterly (every 3 months)

### Test Procedures

1. **Backup Restoration Test**
   ```bash
   # Test in staging environment
   ./scripts/test_backup_restore.sh
   ```

2. **Failover Test**
   ```bash
   # Simulate failure and measure recovery time
   ./scripts/test_failover.sh
   ```

3. **Data Integrity Test**
   ```bash
   # Verify data after restoration
   ./scripts/verify_data_integrity.sh
   ```

---

## 📊 Monitoring & Alerts

### Critical Alerts

Set up alerts for:

- Service downtime > 5 minutes
- Database errors
- Disk space < 10%
- Failed backups
- Health check failures
- High error rates

### Monitoring Dashboard

Access at: `http://your-monitoring-url/cognitionsim`

Key metrics to monitor:
- Service uptime
- Response time
- Error rate
- Database size
- Backup success rate
- Model loading time

---

## 🔐 Security Considerations

1. **Backup Encryption**: All backups should be encrypted at rest
2. **Access Control**: Limit DR access to authorized personnel only
3. **Audit Logs**: Maintain logs of all DR operations
4. **Secret Management**: Store secrets in secure vault (not in backups)

---

## 📞 Escalation Procedures

**Level 1 (< 1 hour)**:
- On-call engineer attempts recovery
- Follow documented procedures

**Level 2 (1-2 hours)**:
- Escalate to DevOps Lead
- Assemble incident response team

**Level 3 (> 2 hours)**:
- Escalate to management
- Consider declaring major incident
- Activate full DR team

---

## 🔄 Continuous Improvement

After each incident:

1. **Post-Mortem Meeting** (within 48 hours)
2. **Document lessons learned**
3. **Update procedures**
4. **Improve automation**
5. **Test new procedures**

---

## 📚 Related Documents

- [PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md) - Production deployment
- [ML_PRODUCTION_CHECKLIST.md](ML_PRODUCTION_CHECKLIST.md) - ML-specific production
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment procedures
- [README.md](README.md) - System overview

---

## ✅ DR Validation

Last Tested: ________________  
Tested By: ________________  
Test Result: ☐ Pass ☐ Fail  
Notes: ________________

---

**Document Version**: 1.0  
**Last Updated**: December 21, 2025  
**Owner**: DevOps Team  
**Review Schedule**: Quarterly
