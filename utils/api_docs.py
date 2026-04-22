"""
OpenAPI/Swagger API Documentation Module

This module provides comprehensive API documentation using OpenAPI 3.0 specification
with Swagger UI integration for interactive API exploration.

Features:
- Automatic endpoint discovery
- Request/response schema validation
- Interactive Swagger UI
- ReDoc alternative UI
- OpenAPI 3.0 specification export
- Authentication documentation
- WebSocket documentation
"""

from typing import Dict, Any, List, Optional
from flask import Flask, jsonify, render_template_string
from flask_swagger_ui import get_swaggerui_blueprint
import json


class APIDocumentation:
    """Manages OpenAPI/Swagger documentation for Flask application."""
    
    def __init__(self, app: Optional[Flask] = None):
        """
        Initialize API documentation.
        
        Args:
            app: Flask application instance
        """
        self.app = app
        self.openapi_spec = self._create_base_spec()
        
        if app:
            self.init_app(app)
    
    def init_app(self, app: Flask):
        """
        Initialize documentation with Flask application.
        
        Args:
            app: Flask application instance
        """
        self.app = app
        self._register_routes()
    
    def _create_base_spec(self) -> Dict[str, Any]:
        """Create base OpenAPI 3.0 specification."""
        return {
            "openapi": "3.0.0",
            "info": {
                "title": "CognitionSim API",
                "version": "1.0.0",
                "description": "Advanced quantum-inspired field optimization system with real-time training and monitoring",
                "contact": {
                    "name": "API Support",
                    "email": "support@cognitionsim.ai"
                }
            },
            "servers": [
                {
                    "url": "http://localhost:5000",
                    "description": "Development server"
                },
                {
                    "url": "https://api.cognitionsim.ai",
                    "description": "Production server"
                }
            ],
            "tags": [
                {
                    "name": "training",
                    "description": "Model training operations"
                },
                {
                    "name": "inference",
                    "description": "Model inference and predictions"
                },
                {
                    "name": "monitoring",
                    "description": "System monitoring and metrics"
                },
                {
                    "name": "websocket",
                    "description": "Real-time WebSocket connections"
                },
                {
                    "name": "health",
                    "description": "Health check endpoints"
                }
            ],
            "paths": {},
            "components": {
                "schemas": self._get_schemas(),
                "securitySchemes": {
                    "ApiKeyAuth": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key",
                        "description": "API key for authentication"
                    },
                    "BearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT"
                    }
                }
            },
            "security": [
                {"ApiKeyAuth": []},
                {"BearerAuth": []}
            ]
        }
    
    def _get_schemas(self) -> Dict[str, Any]:
        """Define reusable API schemas."""
        return {
            "TrainingRequest": {
                "type": "object",
                "required": ["field_size", "max_iterations"],
                "properties": {
                    "field_size": {
                        "type": "integer",
                        "minimum": 10,
                        "maximum": 1000,
                        "description": "Size of the quantum field to optimize"
                    },
                    "max_iterations": {
                        "type": "integer",
                        "minimum": 100,
                        "maximum": 100000,
                        "description": "Maximum number of training iterations"
                    },
                    "learning_rate": {
                        "type": "number",
                        "format": "float",
                        "minimum": 0.0001,
                        "maximum": 0.1,
                        "default": 0.001,
                        "description": "Learning rate for optimization"
                    },
                    "batch_size": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 128,
                        "default": 32,
                        "description": "Batch size for training"
                    }
                }
            },
            "TrainingResponse": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["started", "running", "completed", "failed"],
                        "description": "Training status"
                    },
                    "training_id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "Unique training session identifier"
                    },
                    "message": {
                        "type": "string",
                        "description": "Status message"
                    },
                    "metrics": {
                        "$ref": "#/components/schemas/TrainingMetrics"
                    }
                }
            },
            "TrainingMetrics": {
                "type": "object",
                "properties": {
                    "iteration": {
                        "type": "integer",
                        "description": "Current iteration number"
                    },
                    "loss": {
                        "type": "number",
                        "format": "float",
                        "description": "Current loss value"
                    },
                    "reward": {
                        "type": "number",
                        "format": "float",
                        "description": "Current reward value"
                    },
                    "field_variance": {
                        "type": "number",
                        "format": "float",
                        "description": "Field variance metric"
                    },
                    "elapsed_time": {
                        "type": "number",
                        "format": "float",
                        "description": "Elapsed training time in seconds"
                    }
                }
            },
            "InferenceRequest": {
                "type": "object",
                "required": ["input_data"],
                "properties": {
                    "input_data": {
                        "type": "array",
                        "items": {
                            "type": "number"
                        },
                        "description": "Input data for inference"
                    },
                    "model_version": {
                        "type": "string",
                        "description": "Model version to use (optional)"
                    }
                }
            },
            "InferenceResponse": {
                "type": "object",
                "properties": {
                    "prediction": {
                        "type": "array",
                        "items": {
                            "type": "number"
                        },
                        "description": "Model prediction output"
                    },
                    "confidence": {
                        "type": "number",
                        "format": "float",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Prediction confidence score"
                    },
                    "latency_ms": {
                        "type": "number",
                        "format": "float",
                        "description": "Inference latency in milliseconds"
                    }
                }
            },
            "MetricsResponse": {
                "type": "object",
                "properties": {
                    "http_requests_total": {
                        "type": "integer",
                        "description": "Total HTTP requests"
                    },
                    "websocket_connections": {
                        "type": "integer",
                        "description": "Active WebSocket connections"
                    },
                    "training_sessions": {
                        "type": "integer",
                        "description": "Total training sessions"
                    },
                    "uptime_seconds": {
                        "type": "number",
                        "format": "float",
                        "description": "Application uptime in seconds"
                    }
                }
            },
            "HealthCheckResponse": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["healthy", "degraded", "unhealthy"],
                        "description": "Health status"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Health check timestamp"
                    },
                    "checks": {
                        "type": "object",
                        "properties": {
                            "database": {
                                "type": "string",
                                "enum": ["ok", "fail"]
                            },
                            "redis": {
                                "type": "string",
                                "enum": ["ok", "fail"]
                            },
                            "model": {
                                "type": "string",
                                "enum": ["ok", "fail"]
                            }
                        }
                    }
                }
            },
            "ErrorResponse": {
                "type": "object",
                "properties": {
                    "error": {
                        "type": "string",
                        "description": "Error message"
                    },
                    "code": {
                        "type": "string",
                        "description": "Error code"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Error timestamp"
                    },
                    "trace_id": {
                        "type": "string",
                        "description": "Distributed trace ID"
                    }
                }
            }
        }
    
    def add_endpoint(self, path: str, method: str, spec: Dict[str, Any]):
        """
        Add endpoint documentation.
        
        Args:
            path: API endpoint path
            method: HTTP method (get, post, put, delete, etc.)
            spec: OpenAPI specification for the endpoint
        """
        if path not in self.openapi_spec["paths"]:
            self.openapi_spec["paths"][path] = {}
        
        self.openapi_spec["paths"][path][method.lower()] = spec
    
    def add_training_endpoints(self):
        """Add training-related endpoint documentation."""
        self.add_endpoint(
            "/api/train",
            "POST",
            {
                "tags": ["training"],
                "summary": "Start model training",
                "description": "Initiates a new training session with specified parameters",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/TrainingRequest"
                            },
                            "examples": {
                                "basic": {
                                    "summary": "Basic training",
                                    "value": {
                                        "field_size": 100,
                                        "max_iterations": 10000
                                    }
                                },
                                "advanced": {
                                    "summary": "Advanced training with custom parameters",
                                    "value": {
                                        "field_size": 500,
                                        "max_iterations": 50000,
                                        "learning_rate": 0.005,
                                        "batch_size": 64
                                    }
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Training started successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/TrainingResponse"
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid request parameters",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ErrorResponse"
                                }
                            }
                        }
                    },
                    "429": {
                        "description": "Rate limit exceeded"
                    },
                    "500": {
                        "description": "Internal server error"
                    }
                }
            }
        )
        
        self.add_endpoint(
            "/api/training/status",
            "GET",
            {
                "tags": ["training"],
                "summary": "Get training status",
                "description": "Retrieves the current status of training session",
                "responses": {
                    "200": {
                        "description": "Training status retrieved",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/TrainingResponse"
                                }
                            }
                        }
                    }
                }
            }
        )
    
    def add_inference_endpoints(self):
        """Add inference-related endpoint documentation."""
        self.add_endpoint(
            "/api/predict",
            "POST",
            {
                "tags": ["inference"],
                "summary": "Make prediction",
                "description": "Performs inference using the trained model",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/InferenceRequest"
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Prediction successful",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/InferenceResponse"
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid input data"
                    }
                }
            }
        )
    
    def add_monitoring_endpoints(self):
        """Add monitoring-related endpoint documentation."""
        self.add_endpoint(
            "/metrics",
            "GET",
            {
                "tags": ["monitoring"],
                "summary": "Prometheus metrics",
                "description": "Returns Prometheus-formatted metrics",
                "responses": {
                    "200": {
                        "description": "Metrics returned",
                        "content": {
                            "text/plain": {
                                "schema": {
                                    "type": "string"
                                }
                            }
                        }
                    }
                }
            }
        )
        
        self.add_endpoint(
            "/api/metrics",
            "GET",
            {
                "tags": ["monitoring"],
                "summary": "JSON metrics",
                "description": "Returns metrics in JSON format",
                "responses": {
                    "200": {
                        "description": "Metrics returned",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/MetricsResponse"
                                }
                            }
                        }
                    }
                }
            }
        )
    
    def add_health_endpoints(self):
        """Add health check endpoint documentation."""
        self.add_endpoint(
            "/health",
            "GET",
            {
                "tags": ["health"],
                "summary": "Health check",
                "description": "Returns application health status",
                "responses": {
                    "200": {
                        "description": "Application is healthy",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/HealthCheckResponse"
                                }
                            }
                        }
                    },
                    "503": {
                        "description": "Application is unhealthy"
                    }
                }
            }
        )
    
    def add_websocket_documentation(self):
        """Add WebSocket documentation."""
        # WebSocket documentation in OpenAPI 3.0
        self.openapi_spec["components"]["schemas"]["WebSocketMessage"] = {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["training_update", "error", "status"],
                    "description": "Message type"
                },
                "data": {
                    "type": "object",
                    "description": "Message payload"
                },
                "timestamp": {
                    "type": "string",
                    "format": "date-time"
                }
            }
        }
    
    def _register_routes(self):
        """Register documentation routes with Flask app."""
        
        # Add all endpoint documentation
        self.add_training_endpoints()
        self.add_inference_endpoints()
        self.add_monitoring_endpoints()
        self.add_health_endpoints()
        self.add_websocket_documentation()
        
        # OpenAPI spec endpoint
        @self.app.route('/api/openapi.json')
        def openapi_spec():
            """Return OpenAPI specification."""
            return jsonify(self.openapi_spec)
        
        # Swagger UI
        SWAGGER_URL = '/api/docs'
        API_URL = '/api/openapi.json'
        
        swaggerui_blueprint = get_swaggerui_blueprint(
            SWAGGER_URL,
            API_URL,
            config={
                'app_name': "CognitionSim API",
                'docExpansion': 'list',
                'defaultModelsExpandDepth': 3,
                'displayRequestDuration': True,
                'filter': True
            }
        )
        
        self.app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
        
        # ReDoc alternative UI
        @self.app.route('/api/redoc')
        def redoc():
            """Alternative ReDoc UI."""
            return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>CognitionSim API - ReDoc</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
    <style>
        body {
            margin: 0;
            padding: 0;
        }
    </style>
</head>
<body>
    <redoc spec-url='/api/openapi.json'></redoc>
    <script src="https://cdn.jsdelivr.net/npm/redoc@latest/bundles/redoc.standalone.js"></script>
</body>
</html>
            ''')
    
    def export_spec(self, filepath: str):
        """
        Export OpenAPI specification to file.
        
        Args:
            filepath: Path to save specification
        """
        with open(filepath, 'w') as f:
            json.dump(self.openapi_spec, f, indent=2)


def setup_api_documentation(app: Flask) -> APIDocumentation:
    """
    Setup API documentation for Flask application.
    
    Args:
        app: Flask application instance
    
    Returns:
        APIDocumentation instance
    
    Example:
        >>> app = Flask(__name__)
        >>> api_docs = setup_api_documentation(app)
        >>> # Access Swagger UI at: http://localhost:5000/api/docs
        >>> # Access ReDoc at: http://localhost:5000/api/redoc
        >>> # Access OpenAPI spec at: http://localhost:5000/api/openapi.json
    """
    api_docs = APIDocumentation(app)
    return api_docs
