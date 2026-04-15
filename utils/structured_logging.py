"""
Structured Logging Configuration for ELK/Loki Integration
JSON-formatted logs for log aggregation systems
"""
import logging
import sys
from pythonjsonlogger import jsonlogger
from pathlib import Path
from typing import Optional
import os

# ============================================================
# STRUCTURED LOGGER SETUP
# ============================================================

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter with additional fields for ELK/Loki
    """
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        
        # Add custom fields
        log_record['timestamp'] = self.formatTime(record, self.datefmt)
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        # Add application context
        log_record['application'] = 'cognitionsim-ai'
        log_record['environment'] = os.getenv('FLASK_ENV', 'development')
        
        # Add process info
        log_record['process_id'] = record.process
        log_record['thread_id'] = record.thread
        log_record['thread_name'] = record.threadName


def setup_structured_logging(
    log_level: str = 'INFO',
    logs_dir: Optional[Path] = None,
    enable_console: bool = True,
    enable_file: bool = True
) -> logging.Logger:
    """
    Configure structured logging for ELK/Loki integration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logs_dir: Directory for log files
        enable_console: Enable console logging (JSON format)
        enable_file: Enable file logging (JSON format)
    
    Returns:
        Configured root logger
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = CustomJsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S.%fZ'
    )
    
    # Console handler (JSON)
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler (JSON)
    if enable_file and logs_dir:
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Application log
        app_log_file = logs_dir / 'atlantean_hrm.json.log'
        file_handler = logging.FileHandler(app_log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Error log
        error_log_file = logs_dir / 'atlantean_hrm_error.json.log'
        error_handler = logging.FileHandler(error_log_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
    
    return root_logger


# ============================================================
# LOGSTASH/FLUENTD CONFIGURATION
# ============================================================

LOGSTASH_CONFIG = """
# Logstash Configuration for CognitionSim
# Place in: /etc/logstash/conf.d/cognitionsim.conf

input {
  file {
    path => "/app/logs/atlantean_hrm.json.log"
    codec => "json"
    type => "cognitionsim-app"
    start_position => "beginning"
  }
  
  file {
    path => "/app/logs/atlantean_hrm_error.json.log"
    codec => "json"
    type => "cognitionsim-error"
    start_position => "beginning"
  }
}

filter {
  # Parse timestamp
  date {
    match => [ "timestamp", "ISO8601" ]
    target => "@timestamp"
  }
  
  # Add tags
  mutate {
    add_field => { "application" => "cognitionsim-ai" }
    add_tag => [ "python", "ml", "cognitionsim" ]
  }
  
  # Extract error details if present
  if [exc_info] {
    mutate {
      add_tag => [ "exception" ]
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "cognitionsim-%{+YYYY.MM.dd}"
    document_type => "_doc"
  }
  
  # Optional: stdout for debugging
  # stdout { codec => rubydebug }
}
"""

PROMTAIL_CONFIG = """
# Promtail Configuration for Loki
# Place in: /etc/promtail/config.yml

server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: cognitionsim
    static_configs:
      - targets:
          - localhost
        labels:
          job: cognitionsim
          application: cognitionsim-ai
          __path__: /app/logs/atlantean_hrm.json.log
    
    pipeline_stages:
      - json:
          expressions:
            level: level
            timestamp: timestamp
            message: message
            logger: logger
            module: module
            function: function
      
      - labels:
          level:
          logger:
          module:
      
      - timestamp:
          source: timestamp
          format: RFC3339Nano
      
      - output:
          source: message
"""

FLUENTD_CONFIG = """
# Fluentd Configuration
# Place in: /etc/fluent/fluent.conf

<source>
  @type tail
  path /app/logs/atlantean_hrm.json.log
  pos_file /var/log/td-agent/atlantean_hrm.pos
  tag quadra.matrix.app
  <parse>
    @type json
    time_key timestamp
    time_format %Y-%m-%dT%H:%M:%S.%NZ
  </parse>
</source>

<source>
  @type tail
  path /app/logs/atlantean_hrm_error.json.log
  pos_file /var/log/td-agent/atlantean_hrm_error.pos
  tag quadra.matrix.error
  <parse>
    @type json
    time_key timestamp
    time_format %Y-%m-%dT%H:%M:%S.%NZ
  </parse>
</source>

<filter quadra.matrix.**>
  @type record_transformer
  <record>
    application cognitionsim-ai
    environment "#{ENV['FLASK_ENV'] || 'production'}"
  </record>
</filter>

<match quadra.matrix.**>
  @type elasticsearch
  host elasticsearch
  port 9200
  logstash_format true
  logstash_prefix cognitionsim
  include_tag_key true
  tag_key @log_name
  flush_interval 10s
</match>
"""

# ============================================================
# DOCKER COMPOSE ADDITIONS FOR ELK STACK
# ============================================================

DOCKER_COMPOSE_ELK = """
# Add to docker-compose.yml for ELK stack

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    container_name: elasticsearch
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data
    networks:
      - quadra-network

  logstash:
    image: docker.elastic.co/logstash/logstash:8.11.0
    container_name: logstash
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline
      - ./logs:/app/logs:ro
    ports:
      - "5044:5044"
    environment:
      - "LS_JAVA_OPTS=-Xms256m -Xmx256m"
    depends_on:
      - elasticsearch
    networks:
      - quadra-network

  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    container_name: kibana
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch
    networks:
      - quadra-network

volumes:
  elasticsearch-data:
    driver: local
"""

DOCKER_COMPOSE_LOKI = """
# Add to docker-compose.yml for Loki stack

  loki:
    image: grafana/loki:2.9.0
    container_name: loki
    ports:
      - "3100:3100"
    command: -config.file=/etc/loki/local-config.yaml
    networks:
      - quadra-network

  promtail:
    image: grafana/promtail:2.9.0
    container_name: promtail
    volumes:
      - ./logs:/app/logs:ro
      - ./promtail-config.yml:/etc/promtail/config.yml:ro
    command: -config.file=/etc/promtail/config.yml
    depends_on:
      - loki
    networks:
      - quadra-network

  grafana:
    image: grafana/grafana:10.2.0
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    depends_on:
      - loki
    networks:
      - quadra-network

volumes:
  grafana-data:
    driver: local
"""


def save_log_aggregation_configs(output_dir: Path):
    """Save log aggregation configuration files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Logstash config
    (output_dir / 'logstash.conf').write_text(LOGSTASH_CONFIG)
    
    # Save Promtail config
    (output_dir / 'promtail-config.yml').write_text(PROMTAIL_CONFIG)
    
    # Save Fluentd config
    (output_dir / 'fluent.conf').write_text(FLUENTD_CONFIG)
    
    # Save Docker Compose additions
    (output_dir / 'docker-compose-elk.yml').write_text(DOCKER_COMPOSE_ELK)
    (output_dir / 'docker-compose-loki.yml').write_text(DOCKER_COMPOSE_LOKI)
    
    print(f"✅ Log aggregation configs saved to {output_dir}")
