# Deployment Guide

This guide provides detailed instructions for deploying the Enterprise RAG Q&A System in various environments.

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Platforms](#cloud-platforms)
4. [Production Best Practices](#production-best-practices)
5. [Monitoring & Maintenance](#monitoring--maintenance)

## Local Development

### Quick Start

```bash
# Make the run script executable
chmod +x run.sh

# Run the application
./run.sh
```

### Manual Setup

1. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env and set GOOGLE_API_KEY
   ```

4. **Run Application**
   ```bash
   streamlit run app.py
   ```

5. **Access Application**
   - Open browser to `http://localhost:8501`

### Development Mode Options

**Custom Port:**
```bash
streamlit run app.py --server.port 8080
```

**Enable Debug:**
```bash
STREAMLIT_LOG_LEVEL=debug streamlit run app.py
```

**Disable CORS (for local API access):**
```bash
streamlit run app.py --server.enableCORS false
```

## Docker Deployment

### Basic Docker

1. **Build Image**
   ```bash
   docker build -t streamlit-rag:latest .
   ```

2. **Run Container**
   ```bash
   docker run -d \
     --name rag-app \
     -p 8501:8501 \
     -e GOOGLE_API_KEY=your_api_key \
     -v $(pwd)/chroma_db:/app/chroma_db \
     streamlit-rag:latest
   ```

3. **View Logs**
   ```bash
   docker logs -f rag-app
   ```

4. **Stop Container**
   ```bash
   docker stop rag-app
   docker rm rag-app
   ```

### Docker Compose

1. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

2. **Start Services**
   ```bash
   docker-compose up -d
   ```

3. **View Logs**
   ```bash
   docker-compose logs -f
   ```

4. **Stop Services**
   ```bash
   docker-compose down
   ```

5. **Rebuild After Changes**
   ```bash
   docker-compose up -d --build
   ```

### Multi-Stage Docker Build (Optimized)

Create `Dockerfile.optimized`:

```dockerfile
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

Build and run:
```bash
docker build -f Dockerfile.optimized -t streamlit-rag:optimized .
docker run -d -p 8501:8501 --env-file .env streamlit-rag:optimized
```

## Cloud Platforms

### Streamlit Community Cloud

1. **Prepare Repository**
   - Ensure code is in GitHub repository
   - Verify `requirements.txt` is present
   - Add `.streamlit/secrets.toml` to `.gitignore`

2. **Deploy**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select repository, branch, and `app.py`

3. **Configure Secrets**
   - In app settings, add secrets:
   ```toml
   GOOGLE_API_KEY = "your_api_key_here"
   ENVIRONMENT = "production"
   ```

4. **Deploy**
   - Click "Deploy"
   - App will be available at `https://[app-name].streamlit.app`

### AWS Deployment

#### Option 1: AWS App Runner

1. **Create ECR Repository**
   ```bash
   aws ecr create-repository --repository-name streamlit-rag
   ```

2. **Build and Push Image**
   ```bash
   aws ecr get-login-password --region us-east-1 | \
     docker login --username AWS --password-stdin [account-id].dkr.ecr.us-east-1.amazonaws.com
   
   docker build -t streamlit-rag .
   docker tag streamlit-rag:latest [account-id].dkr.ecr.us-east-1.amazonaws.com/streamlit-rag:latest
   docker push [account-id].dkr.ecr.us-east-1.amazonaws.com/streamlit-rag:latest
   ```

3. **Create App Runner Service**
   ```bash
   aws apprunner create-service \
     --service-name streamlit-rag \
     --source-configuration '{
       "ImageRepository": {
         "ImageIdentifier": "[account-id].dkr.ecr.us-east-1.amazonaws.com/streamlit-rag:latest",
         "ImageRepositoryType": "ECR"
       }
     }' \
     --instance-configuration '{
       "Cpu": "1 vCPU",
       "Memory": "2 GB"
     }'
   ```

#### Option 2: ECS Fargate

1. **Create Task Definition** (`task-definition.json`)
   ```json
   {
     "family": "streamlit-rag",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "1024",
     "memory": "2048",
     "containerDefinitions": [{
       "name": "streamlit-rag",
       "image": "[account-id].dkr.ecr.us-east-1.amazonaws.com/streamlit-rag:latest",
       "portMappings": [{
         "containerPort": 8501,
         "protocol": "tcp"
       }],
       "environment": [
         {"name": "ENVIRONMENT", "value": "production"}
       ],
       "secrets": [
         {
           "name": "GOOGLE_API_KEY",
           "valueFrom": "arn:aws:secretsmanager:us-east-1:[account-id]:secret:google-api-key"
         }
       ]
     }]
   }
   ```

2. **Deploy to ECS**
   ```bash
   aws ecs create-service \
     --cluster default \
     --service-name streamlit-rag \
     --task-definition streamlit-rag \
     --desired-count 1 \
     --launch-type FARGATE \
     --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
   ```

### Google Cloud Platform

#### Cloud Run Deployment

1. **Enable Required APIs**
   ```bash
   gcloud services enable run.googleapis.com containerregistry.googleapis.com
   ```

2. **Build and Push to GCR**
   ```bash
   gcloud builds submit --tag gcr.io/[project-id]/streamlit-rag
   ```

3. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy streamlit-rag \
     --image gcr.io/[project-id]/streamlit-rag \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars ENVIRONMENT=production \
     --set-secrets GOOGLE_API_KEY=google-api-key:latest
   ```

4. **Access Application**
   - Service URL will be provided in output
   - Example: `https://streamlit-rag-xxx-uc.a.run.app`

### Azure

#### Azure Container Instances

1. **Create Resource Group**
   ```bash
   az group create --name streamlit-rag-rg --location eastus
   ```

2. **Create Container Registry**
   ```bash
   az acr create --resource-group streamlit-rag-rg \
     --name streamlitragacr --sku Basic
   ```

3. **Build and Push Image**
   ```bash
   az acr build --registry streamlitragacr \
     --image streamlit-rag:latest .
   ```

4. **Deploy Container**
   ```bash
   az container create \
     --resource-group streamlit-rag-rg \
     --name streamlit-rag \
     --image streamlitragacr.azurecr.io/streamlit-rag:latest \
     --dns-name-label streamlit-rag-app \
     --ports 8501 \
     --environment-variables ENVIRONMENT=production \
     --secure-environment-variables GOOGLE_API_KEY=your_api_key
   ```

### Kubernetes

1. **Create Deployment** (`k8s-deployment.yaml`)
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: streamlit-rag
   spec:
     replicas: 2
     selector:
       matchLabels:
         app: streamlit-rag
     template:
       metadata:
         labels:
           app: streamlit-rag
       spec:
         containers:
         - name: streamlit-rag
           image: streamlit-rag:latest
           ports:
           - containerPort: 8501
           env:
           - name: ENVIRONMENT
             value: "production"
           - name: GOOGLE_API_KEY
             valueFrom:
               secretKeyRef:
                 name: google-api-key
                 key: api-key
           resources:
             requests:
               memory: "1Gi"
               cpu: "500m"
             limits:
               memory: "2Gi"
               cpu: "1000m"
   ---
   apiVersion: v1
   kind: Service
   metadata:
     name: streamlit-rag-service
   spec:
     selector:
       app: streamlit-rag
     ports:
     - protocol: TCP
       port: 80
       targetPort: 8501
     type: LoadBalancer
   ```

2. **Create Secret**
   ```bash
   kubectl create secret generic google-api-key \
     --from-literal=api-key=your_google_api_key
   ```

3. **Deploy**
   ```bash
   kubectl apply -f k8s-deployment.yaml
   ```

4. **Get Service URL**
   ```bash
   kubectl get service streamlit-rag-service
   ```

## Production Best Practices

### Security

1. **Environment Variables**
   - Never commit `.env` file
   - Use secret management (AWS Secrets Manager, GCP Secret Manager)
   - Rotate API keys regularly

2. **Authentication**
   - Enable `ENABLE_AUTH=true` in production
   - Implement proper authentication (OAuth, SAML)
   - Use HTTPS only

3. **Network Security**
   - Use firewall rules
   - Implement rate limiting
   - Enable DDoS protection

### Performance

1. **Resource Allocation**
   - Minimum: 1 CPU, 2GB RAM
   - Recommended: 2 CPU, 4GB RAM
   - Scale based on load

2. **Caching**
   - Use Streamlit's `@st.cache_data` decorator
   - Implement Redis for distributed caching
   - Cache embeddings and vector store

3. **Load Balancing**
   - Use application load balancer
   - Enable health checks
   - Configure auto-scaling

### Monitoring

1. **Application Metrics**
   - Response time
   - Error rate
   - Query volume
   - Resource utilization

2. **Logging**
   - Centralized logging (CloudWatch, Stackdriver)
   - Log levels: INFO, WARNING, ERROR
   - Include request IDs

3. **Alerting**
   - Set up alerts for errors
   - Monitor API rate limits
   - Track resource thresholds

## Monitoring & Maintenance

### Health Checks

Add to your deployment:

```python
# health_check.py
import requests
import sys

def check_health():
    try:
        response = requests.get("http://localhost:8501/_stcore/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            sys.exit(0)
        else:
            print("❌ Health check failed")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Health check error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_health()
```

### Backup Strategy

1. **Vector Store Backup**
   ```bash
   # Backup chroma_db
   tar -czf chroma_db_backup_$(date +%Y%m%d).tar.gz chroma_db/
   
   # Upload to S3
   aws s3 cp chroma_db_backup_$(date +%Y%m%d).tar.gz s3://backups/
   ```

2. **Restore from Backup**
   ```bash
   # Download from S3
   aws s3 cp s3://backups/chroma_db_backup_20240101.tar.gz .
   
   # Extract
   tar -xzf chroma_db_backup_20240101.tar.gz
   ```

### Update Strategy

1. **Rolling Update**
   ```bash
   # Build new version
   docker build -t streamlit-rag:v2 .
   
   # Tag as latest
   docker tag streamlit-rag:v2 streamlit-rag:latest
   
   # Update deployment
   kubectl set image deployment/streamlit-rag streamlit-rag=streamlit-rag:v2
   ```

2. **Blue-Green Deployment**
   - Deploy new version alongside old
   - Test new version
   - Switch traffic
   - Keep old version for rollback

### Troubleshooting

**Check Logs:**
```bash
# Docker
docker logs rag-app

# Kubernetes
kubectl logs -f deployment/streamlit-rag

# Cloud Run
gcloud run services logs read streamlit-rag
```

**Common Issues:**

1. **Out of Memory**
   - Increase container memory
   - Reduce batch sizes
   - Optimize vector store

2. **API Rate Limits**
   - Implement exponential backoff
   - Use caching
   - Request quota increase

3. **Slow Response**
   - Check network latency
   - Optimize retrieval parameters
   - Add read replicas

## Support

For deployment issues:
- Check application logs
- Review configuration
- Verify API credentials
- Check network connectivity

For further assistance, create an issue in the repository.
