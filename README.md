# 🍷 Wine Classifier ML Project - Kubernetes Deployment

**Ek complete Machine Learning project jo Kubernetes par deploy ho sake**

## Project Overview

Yeh project ek **Wine Cultivar Classification** ML model banata hai jo wine ke chemical properties dekh kar predict karta hai ki wine kis class (cultivar) ki hai.

### Features
- ✅ Scikit-learn RandomForest model (Wine dataset par trained)
- ✅ FastAPI based REST API with automatic Swagger docs
- ✅ Prometheus metrics support
- ✅ Production-ready Kubernetes manifests (Deployment, Service, HPA)
- ✅ Health checks (Liveness & Readiness probes)
- ✅ Docker multi-stage build (optimized image)
- ✅ Non-root user security
- ✅ Proper logging and error handling

---

## 📁 Project Structure

```
ml-k8s-project/
├── README.md
├── requirements.txt
├── train.py              # Model training script
├── app.py                # FastAPI inference service
├── Dockerfile            # Multi-stage Docker build
├── k8s/
│   ├── deployment.yaml   # Kubernetes Deployment
│   ├── service.yaml      # Kubernetes Service
│   └── hpa.yaml          # Horizontal Pod Autoscaler
└── models/               # Trained model (after running train.py)
```

---

## 🚀 Step-by-Step Setup

### Step 1: Local Setup & Training

```bash
# Clone ya download karo project
cd ml-k8s-project

# Virtual environment banao (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Dependencies install karo
pip install -r requirements.txt

# Model train karo (yeh models/ folder mein .pkl file banayega)
python train.py
```

Training ke baad `models/wine_classifier.pkl` aur `models/metadata.json` ban jayega.

### Step 2: Local Testing (FastAPI)

```bash
# Run the API locally
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Browser mein jaao: http://localhost:8000/docs

Wahan se **/predict** endpoint try kar sakte ho.

### Step 3: Docker Build & Run

```bash
# Docker image build karo
docker build -t wine-classifier-api:latest .

# Local Docker container run karo (test ke liye)
docker run -d -p 8000:8000 --name wine-api wine-classifier-api:latest

# Test karo
curl http://localhost:8000/health
```

### Step 4: Kubernetes Deployment

#### Prerequisites
- Kubernetes cluster (Minikube, Kind, Docker Desktop, ya EKS/AKS/GKE)
- kubectl configured

#### Deploy karne ke steps:

```bash
# 1. Docker image ko registry mein push karo (Docker Hub, ECR, GCR etc.)
# Example (Docker Hub):
docker tag wine-classifier-api:latest yourusername/wine-classifier-api:v1
docker push yourusername/wine-classifier-api:v1

# 2. deployment.yaml mein image name change karo
#    (k8s/deployment.yaml file edit karo - "your-docker-registry/..." wali line)

# 3. Kubernetes resources deploy karo
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# 4. Check karo pods running hain ya nahi
kubectl get pods -l app=wine-classifier
kubectl get svc wine-classifier-service
kubectl get hpa wine-classifier-hpa

# 5. Port-forward for local testing
kubectl port-forward svc/wine-classifier-service 8000:80
```

Ab browser mein `http://localhost:8000/docs` kholo.

---

## 📡 API Endpoints

| Endpoint       | Method | Description                     |
|----------------|--------|---------------------------------|
| `/`            | GET    | Welcome message                 |
| `/health`      | GET    | Health check (for K8s probes)   |
| `/predict`     | POST   | Wine class prediction           |
| `/model-info`  | GET    | Model metadata                  |
| `/metrics`     | GET    | Prometheus metrics              |
| `/docs`        | GET    | Swagger UI (interactive docs)   |

### Example Prediction Request

```json
POST /predict
{
  "alcohol": 13.72,
  "malic_acid": 1.43,
  "ash": 2.50,
  "alcalinity_of_ash": 16.7,
  "magnesium": 108.0,
  "total_phenols": 3.40,
  "flavanoids": 3.67,
  "nonflavanoid_phenols": 0.19,
  "proanthocyanins": 2.04,
  "color_intensity": 6.80,
  "hue": 0.89,
  "od280_od315_of_diluted_wines": 2.87,
  "proline": 1285.0
}
```

**Response:**
```json
{
  "predicted_class": 0,
  "predicted_label": "class_0",
  "confidence": 0.98,
  "probabilities": {
    "class_0": 0.98,
    "class_1": 0.01,
    "class_2": 0.01
  }
}
```

---

## 🔧 Customization Ideas (Advanced)

1. **Model change karna**: `train.py` mein XGBoost, LightGBM ya Neural Network use karo
2. **Different dataset**: `load_breast_cancer()` ya `load_diabetes()` try karo
3. **Model storage**: Model ko S3/GCS pe store karo aur init container se download karo (production best practice)
4. **Authentication**: FastAPI + OAuth2 add karo
5. **CI/CD**: GitHub Actions + ArgoCD pipeline banao
6. **Monitoring**: Grafana + Prometheus dashboard add karo

---

## 🛠️ Troubleshooting

| Problem                    | Solution                                      |
|---------------------------|-----------------------------------------------|
| Model not found           | Pehle `python train.py` run karo             |
| Pod CrashLoopBackOff      | `kubectl logs` dekh lo - model load error ho sakta hai |
| Image pull error          | Image registry mein push hua hai? Secret add kiya? |
| HPA not scaling           | Metrics server installed hai cluster mein?    |

---

## 📚 Tech Stack

- **ML**: scikit-learn, RandomForest
- **API**: FastAPI + Uvicorn + Pydantic
- **Container**: Docker (multi-stage)
- **Orchestration**: Kubernetes (Deployment, Service, HPA)
- **Observability**: Prometheus (via fastapi-instrumentator)
- **Security**: Non-root user, resource limits

---

## 🙏 Credits

Yeh project aapke liye specially banaya gaya hai Kubernetes + MLOps practice ke liye.

Agar kuch changes chahiye (jaise different model, dataset, ya more advanced MLOps with MLflow + Kubeflow), toh batao!

**Happy Deploying! 🚀**# ml-k8s-project
