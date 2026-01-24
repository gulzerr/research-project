# SA-1B-Scale Fog Detection Pipeline

## Project Overview

A distributed cloud-based pipeline for detecting and classifying fog in massive image datasets (11M+ images) using hybrid FADE-based weak labeling and Vision Transformer classification.

### Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Cloud Infrastructure                     │
│  ┌────────────┐    ┌──────────────┐    ┌─────────────────┐ │
│  │  S3/Blob   │───▶│  Node.js     │───▶│   RabbitMQ      │ │
│  │  Storage   │    │  Orchestrator│    │   Message Queue │ │
│  └────────────┘    └──────────────┘    └─────────────────┘ │
│                                               │              │
│                          ┌────────────────────┴──────┐      │
│                          ▼                           ▼      │
│                  ┌───────────────┐         ┌──────────────┐│
│                  │ MATLAB Workers│         │Python ViT    ││
│                  │ (FADE scoring)│         │Classifier    ││
│                  └───────┬───────┘         └──────┬───────┘│
│                          │                        │        │
│                          ▼                        ▼        │
│                  ┌────────────────────────────────────────┐│
│                  │       Results Database / CSV          ││
│                  └────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Pipeline Stages

1. **Image Ingestion**: Load image paths from SA-1B corpus into task queue
2. **FADE Processing**: MATLAB workers compute fog density scores (weak labels)
3. **Weak Label Generation**: Threshold FADE scores to create training data
4. **Classifier Training**: Train ViT on labeled data (ACDC, Foggy Cityscapes, SynFog)
5. **Refinement**: Re-classify images with trained model for precise fog detection
6. **Evaluation**: Compute precision/recall, ROC-AUC, correlation metrics

## Getting Started

### Prerequisites

- Docker & Docker Compose
- MATLAB Runtime (or MATLAB license)
- Node.js 18+
- Python 3.9+
- RabbitMQ
- Cloud storage (AWS S3 / Azure Blob / GCS)

### Quick Start

```bash
# Install dependencies
cd services/nodejs-orchestrator && npm install && cd ../..
cd services/python-classifier && pip install -r requirements.txt && cd ../..

# Start services with Docker Compose
docker-compose up -d

# Submit processing job
node services/nodejs-orchestrator/src/submit-job.js --input s3://my-bucket/sa1b-images/
```

## Project Structure

```
research-project/
├── FADE_release/              # Original MATLAB FADE implementation
├── services/
│   ├── nodejs-orchestrator/   # Job orchestration and RabbitMQ management
│   ├── matlab-worker/         # MATLAB FADE processing workers
│   ├── python-classifier/     # ViT training and inference
│   └── results-aggregator/    # Collect and analyze results
├── datasets/                  # Training data setup scripts
├── docker/                    # Dockerfiles for each service
├── deployment/                # Kubernetes/cloud deployment configs
├── evaluation/                # Metrics and benchmarking scripts
└── docker-compose.yml         # Local development setup
```

## Datasets

- **SA-1B**: 11M images for fog detection inference
- **ACDC**: Adverse conditions (fog split) for supervised training
- **Foggy Cityscapes**: Synthetic fog augmentation for training
- **SynFog**: Additional synthetic fog data
- **RESIDE RTTS & Foggy Driving**: External validation sets

## Methods

### 1. FADE-based Weak Labeling
- Compute referenceless fog density scores using MATLAB FADE algorithm
- Threshold scores: `clear < 0.3`, `light_fog 0.3-0.6`, `dense_fog > 0.6`
- Generate initial training dataset from SA-1B subset

### 2. Vision Transformer Classification
- Architecture: ViT-B/16 or DeiT-Small
- Multi-class output: `{clear, light_fog, dense_fog}`
- Training: Mixed synthetic (Foggy Cityscapes, SynFog) + real (ACDC)
- Data augmentation: random crop, color jitter, horizontal flip
- Class balancing: weighted loss or focal loss

### 3. Calibration & Refinement
- Platt scaling or temperature scaling on validation set
- Map classifier logits to continuous fog density scores
- Ensemble FADE + ViT predictions for robust detection

### 4. Distributed Processing
- RabbitMQ work queue with ~100-1000 workers
- Batch size: 100-500 images per task
- Expected throughput: 1000-5000 images/sec (distributed)
- Total time for 11M images: ~30 mins - 3 hours (depending on workers)

## Evaluation Metrics

- **Detection Quality**: Precision, Recall, F1, ROC-AUC
- **Density Correlation**: Spearman/Pearson correlation between FADE and ViT scores
- **Generalization**: Cross-dataset validation (ACDC→RESIDE, etc.)
- **Scalability**: Images/sec, total wall-clock time, cost per 1M images

## References

1. Kirillov et al., "Segment Anything", arXiv:2304.02643, 2023
2. Choi et al., "FADE/DEFADE", LIVE, UT Austin, 2014
3. Sakaridis et al., "ACDC Dataset", ICCV 2021
4. Sakaridis et al., "Semantic Foggy Scene Understanding", ECCV 2018

---

## Octave Installation (for local testing)

```bash
octave --eval "pkg update"
octave --eval "pkg install -forge image"
octave --eval "pkg install -forge io"
octave --eval "pkg install -forge datatypes"
octave --eval "pkg install -forge statistics"
```
