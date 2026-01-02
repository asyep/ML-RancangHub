# RancangHub ML V2 – AI Construction Cost Estimation System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production-success.svg)]()

RancangHub ML V2 adalah sistem machine learning untuk memprediksi Bill of Quantities (BOQ) konstruksi berdasarkan deskripsi proyek, lokasi, dan periode harga. Model ini menggunakan ensemble Random Forest dan XGBoost yang terintegrasi dengan aplikasi RancangHub (React + Node.js/Express + PostgreSQL).

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running the Model](#running-the-model)
- [Model Training](#model-training)
- [Testing & Validation](#testing--validation)
- [API Integration](#api-integration)
- [Data Quality](#data-quality)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

### Core Capabilities
- **AI-Powered BOQ Prediction**: Prediksi otomatis daftar pekerjaan (AHSP items) dan volume dari deskripsi proyek
- **Ensemble Learning**: Kombinasi Random Forest dan XGBoost dengan weighted blending
- **Real-time Integration**: API endpoint untuk frontend AI Project Wizard
- **Data Quality Guard**: Filtering otomatis untuk memastikan hanya data valid yang digunakan

### Input Parameters
- Nama dan deskripsi proyek
- Lokasi (kabupaten/kota)
- Periode harga
- Volume threshold (opsional)

### Output
- Daftar AHSP items yang relevan
- Prediksi volume untuk setiap item
- Confidence scores
- Unit dan kode AHSP

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     FRONTEND (React)                         │
│                  AI Project Wizard                           │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP API
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              BACKEND (Node.js/Express)                       │
│           API Routes & Business Logic                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
    ┌────────┐   ┌─────────┐   ┌──────────┐
    │Database│   │ML Model │   │Resources │
    │(Postgre│   │(Python) │   │(Prices)  │
    │  SQL)  │   │         │   │          │
    └────────┘   └─────────┘   └──────────┘
```

### ML Pipeline Flow

```
1. Data Extraction (data_extractor.py)
   ↓
2. Feature Engineering (feature_builder.py)
   ↓
3. Model Training (trainer.py)
   ├─ Random Forest
   └─ XGBoost
   ↓
4. Model Evaluation & Versioning
   ↓
5. Deployment (predictor.py)
```

---

## Run Project

# 1. Basic run
uvicorn app.main:app --reload

# 2. Atau dengan custom host & port
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 3. Atau dengan log level
uvicorn app.main:app --reload --log-level debug

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- PostgreSQL 12+
- Node.js 16+ (for backend integration)
- Git

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/ml-rancanghub.git
cd ml-rancanghub
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
joblib>=1.3.0
psycopg2-binary>=2.9.0  # PostgreSQL adapter
python-dotenv>=1.0.0
```

### Step 4: Configure Environment

Create `.env` file in project root:

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_USER=rancanghub
DB_PASSWORD=your_password
DB_NAME=rancanghub_v3

# Model Configuration
MODEL_DIR=./models_v2
MIN_COEFFICIENT_VALUE=0.001

# Optional: Email Notifications
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
NOTIFICATION_EMAIL=recipient@example.com
```

### Step 5: Verify Database Connection

```bash
# Test database connection
python -c "from app.db import get_db_connection; conn = get_db_connection(); print('✓ Database connected'); conn.close()"
```

---

## 🚀 Quick Start

### Complete Workflow (First Time Setup)

```bash
# 1. Activate virtual environment
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# 2. Verify data quality
python -m app.ml_v2.test_data_quality

# 3. Train initial model
python -m app.ml_v2.trainer

# 4. Test prediction
python -m app.ml_v2.test_prediction

# 5. Ready to integrate with backend!
```

Expected timeline:
- Data quality check: ~30 seconds
- Model training: 2-5 minutes (depends on dataset size)
- Testing: ~10 seconds

---

## 🎯 Running the Model

### Method 1: Direct Python Script (Recommended for Testing)

#### A. Using Test Script

```bash
# Edit test_prediction.py first and set valid UUIDs
python -m app.ml_v2.test_prediction
```

#### B. Using Python Interactive

```python
# Open Python shell
python

# Import predictor
from app.ml_v2.predictor import predict_for_new_project

# Run prediction (ganti dengan UUID yang valid)
version, df_results = predict_for_new_project(
    regency_id="550e8400-e29b-41d4-a716-446655440000",
    price_period_id="660e8400-e29b-41d4-a716-446655440000",
    project_name="Pembangunan Gedung Sekolah 2 Lantai",
    description="Gedung sekolah 2 lantai luas 500m2 dengan struktur beton bertulang, 20 ruang kelas",
    volume_threshold=0.1
)

# View results
print(f"Model Version: {version}")
print(f"Total Items: {len(df_results)}")
print(df_results.head(20))

# Save to CSV (optional)
df_results.to_csv("prediction_results.csv", index=False)
```

#### C. Create Custom Script

Create file: `my_prediction.py`

```python
from app.ml_v2.predictor import predict_for_new_project
import sys

def main():
    # Replace with actual UUIDs from your database
    REGENCY_ID = "your-regency-uuid-here"
    PRICE_PERIOD_ID = "your-period-uuid-here"

    # Project details
    project_name = "Pembangunan Gedung Kantor"
    description = "Gedung kantor 3 lantai luas 1000m2"

    try:
        print("Running prediction...")
        version, predictions = predict_for_new_project(
            regency_id=REGENCY_ID,
            price_period_id=PRICE_PERIOD_ID,
            project_name=project_name,
            description=description,
            volume_threshold=0.1
        )

        print(f"\n✓ Model Version: {version}")
        print(f"✓ Predicted {len(predictions)} items")
        print("\nTop 10 Results:")
        print(predictions.head(10).to_string(index=False))

        # Save results
        output_file = f"predictions_{version}.csv"
        predictions.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")

    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

Run it:
```bash
python my_prediction.py
```

---

### Method 2: Via Backend API (Production)

#### A. Start Backend Server

```bash
# Assuming you have the backend setup
cd backend
npm install
npm start  # or npm run dev
```

#### B. Call API Endpoint

**Using cURL:**

```bash
curl -X POST http://localhost:3000/api/ai/predict \
  -H "Content-Type: application/json" \
  -d '{
    "regency_id": "550e8400-e29b-41d4-a716-446655440000",
    "price_period_id": "660e8400-e29b-41d4-a716-446655440000",
    "project_name": "Pembangunan Gedung Sekolah",
    "description": "Gedung 2 lantai 500m2",
    "volume_threshold": 0.1
  }'
```

**Using Postman:**

1. Method: `POST`
2. URL: `http://localhost:3000/api/ai/predict`
3. Headers: `Content-Type: application/json`
4. Body (raw JSON):
```json
{
  "regency_id": "550e8400-e29b-41d4-a716-446655440000",
  "price_period_id": "660e8400-e29b-41d4-a716-446655440000",
  "project_name": "Pembangunan Gedung Sekolah",
  "description": "Gedung 2 lantai 500m2",
  "volume_threshold": 0.1
}
```

**Using JavaScript (Frontend):**

```javascript
const predictProject = async () => {
  const response = await fetch('http://localhost:3000/api/ai/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      regency_id: "550e8400-e29b-41d4-a716-446655440000",
      price_period_id: "660e8400-e29b-41d4-a716-446655440000",
      project_name: "Pembangunan Gedung Sekolah",
      description: "Gedung 2 lantai 500m2",
      volume_threshold: 0.1
    })
  });

  const result = await response.json();
  console.log('Predictions:', result.predictions);
};
```

---

### Method 3: Via Frontend (AI Project Wizard)

1. **Start Full Stack Application:**

```bash
# Terminal 1: Backend
cd backend
npm start

# Terminal 2: Frontend
cd frontend
npm run dev

# Terminal 3: Python ML service (if separate)
python -m app.ml_v2.api_service  # if you have one
```

2. **Access Frontend:**
   - Open browser: `http://localhost:5173` (or your Vite port)
   - Navigate to: **AI Project Wizard** atau **Create Project with AI**

3. **Fill Form:**
   - Project Name: e.g., "Pembangunan Gedung Sekolah"
   - Location: Select from dropdown
   - Price Period: Select from dropdown
   - Description: e.g., "Gedung 2 lantai 500m2"
   - Click: **Generate Predictions** atau **Predict BOQ**

4. **Review Results:**
   - View predicted AHSP items in table
   - Edit/remove items if needed
   - Adjust volumes
   - Click **Create Project** to save

---

### Method 4: Batch Predictions (Multiple Projects)

Create file: `batch_predict.py`

```python
import pandas as pd
from app.ml_v2.predictor import predict_for_new_project

# Load projects from CSV
projects_df = pd.read_csv("projects_to_predict.csv")

all_results = []

for idx, row in projects_df.iterrows():
    print(f"\nProcessing project {idx+1}/{len(projects_df)}: {row['project_name']}")

    try:
        version, predictions = predict_for_new_project(
            regency_id=row['regency_id'],
            price_period_id=row['price_period_id'],
            project_name=row['project_name'],
            description=row['description'],
            volume_threshold=0.1
        )

        # Add project info to results
        predictions['project_name'] = row['project_name']
        predictions['project_id_temp'] = idx + 1

        all_results.append(predictions)
        print(f"  ✓ Predicted {len(predictions)} items")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        continue

# Combine all results
if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv("batch_predictions.csv", index=False)
    print(f"\n✓ All predictions saved to: batch_predictions.csv")
```

Input CSV format (`projects_to_predict.csv`):
```csv
regency_id,price_period_id,project_name,description
550e8400-e29b-41d4-a716-446655440000,660e8400-e29b-41d4-a716-446655440000,Gedung Sekolah,Gedung 2 lantai 500m2
550e8400-e29b-41d4-a716-446655440001,660e8400-e29b-41d4-a716-446655440000,Gedung Kantor,Gedung 3 lantai 1000m2
```

Run:
```bash
python batch_predict.py
```

---

## 🏋️ Model Training

### Data Requirements

Minimum requirements for training:
- **Projects**: ≥5 completed projects
- **Samples**: ≥100 project items
- **AHSP Items**: ≥30 items with valid coefficients

### Training Process

#### 1. Data Quality Check

```bash
python -m app.ml_v2.test_data_quality
```

Expected output:
```
============================================================
DATA QUALITY TEST
MIN_COEFFICIENT_VALUE = 0.001
============================================================

[Test 1] Load all AHSP items...
  Total AHSP items: 55
  Valid (has coefficients >= 0.001): 46 (83.64%)
  Invalid (no/too small coefficients): 9 (16.36%)

[Test 2] Load valid AHSP items...
  Loaded 46 valid AHSP items
  ✓ load_valid_ahsp_items() works correctly

[Test 3] Verify coefficient values (>= 0.001)...
  Total coefficients: 228
  Valid (>= 0.001): 228 (100.00%)
  ✓ All coefficients are valid (>= 0.001)

...

============================================================
ALL TESTS PASSED!
============================================================
```

#### 2. Train Models

```bash
python -m app.ml_v2.trainer
```

Training process:
1. Load and validate data
2. Feature engineering
3. Train/validation split (80/20 by project)
4. Train Random Forest model
5. Train XGBoost model
6. Evaluate both models
7. Calculate ensemble weights
8. Save models and metadata

Output files:
```
models_v2/
├── latest.json                      # Points to current version
├── preprocess_v2-20260102-151043.pkl  # Preprocessing pipeline
├── rf_v2-20260102-151043.pkl          # Random Forest model
├── xgb_v2-20260102-151043.pkl         # XGBoost model
└── meta_v2-20260102-151043.json       # Training metadata & metrics
```

#### 3. Model Metrics

Training output example:
```
============================================================
MODEL TRAINING STARTED
============================================================

[Data Loading]
  Projects: 8
  AHSP Items: 46
  Training samples: 274

[Feature Engineering]
  Numeric features: 5
  Categorical features: 6
  Flag features: 5

[Training Random Forest]
  Train MAE: 42.15
  Valid MAE: 50.89
  Train R²: 0.81
  Valid R²: 0.73

[Training XGBoost]
  Train MAE: 48.31
  Valid MAE: 54.45
  Train R²: 0.75
  Valid R²: 0.69

[Ensemble Performance]
  Weights: RF=0.6, XGB=0.4
  Valid MAE: 52.67
  Valid R²: 0.71

[Model Saved]
  Version: v2-20260102-151043
  Location: models_v2/

============================================================
TRAINING COMPLETED SUCCESSFULLY
============================================================
```

---

## 🧪 Testing & Validation

### 1. Unit Tests (Coming Soon)

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_predictor.py
```

### 2. Manual Validation

#### Test Prediction Accuracy

```python
from app.ml_v2.predictor import predict_for_new_project
from app.db import read_sql_df

# Get an existing project for comparison
sql = "SELECT * FROM projects WHERE status = 'completed' LIMIT 1"
test_project = read_sql_df(sql).iloc[0]

# Predict
version, predictions = predict_for_new_project(
    regency_id=test_project['regency_id'],
    price_period_id=test_project['price_period_id'],
    project_name=test_project['project_name'],
    description=test_project['description'],
    volume_threshold=0.0
)

# Get actual items
sql_actual = f"SELECT * FROM project_items WHERE project_id = '{test_project['id']}'"
actual_items = read_sql_df(sql_actual)

# Compare
predicted_ahsp = set(predictions['ahsp_item_id'])
actual_ahsp = set(actual_items['ahsp_item_id'])

precision = len(predicted_ahsp & actual_ahsp) / len(predicted_ahsp)
recall = len(predicted_ahsp & actual_ahsp) / len(actual_ahsp)

print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
```

---

## 🔌 API Integration

### Backend Integration (Node.js/Express)

Example backend endpoint:

```javascript
// routes/ai.routes.js
const { spawn } = require('child_process');

router.post('/predict', async (req, res) => {
  try {
    const { regency_id, price_period_id, project_name, description } = req.body;

    // Call Python ML service
    const python = spawn('python', [
      '-m', 'app.ml_v2.predictor',
      '--regency_id', regency_id,
      '--price_period_id', price_period_id,
      '--project_name', project_name,
      '--description', description
    ]);

    let dataString = '';

    python.stdout.on('data', (data) => {
      dataString += data.toString();
    });

    python.on('close', (code) => {
      if (code === 0) {
        const result = JSON.parse(dataString);
        res.json({
          success: true,
          version: result.version,
          predictions: result.predictions
        });
      } else {
        res.status(500).json({
          success: false,
          message: 'Prediction failed'
        });
      }
    });

  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});
```

### Frontend Integration (React)

```javascript
// AIProjectWizard.jsx
const predictProject = async (projectData) => {
  try {
    const response = await fetch('/api/ai/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        regency_id: projectData.regencyId,
        price_period_id: projectData.pricePeriodId,
        project_name: projectData.name,
        description: projectData.description
      })
    });

    const result = await response.json();

    if (result.success) {
      setPredictions(result.predictions);
      setModelVersion(result.version);
    }
  } catch (error) {
    console.error('Prediction error:', error);
  }
};
```

---

## 🔍 Data Quality

### Coefficient Validation

The system enforces strict data quality rules:

1. **Valid Coefficient Criteria**:
   - `coefficient IS NOT NULL`
   - `coefficient >= MIN_COEFFICIENT_VALUE` (default: 0.001)

2. **Filtering Process**:
   - Invalid AHSP items excluded from training
   - Only valid items returned in predictions
   - Backend cost calculations always succeed

### Data Quality Metrics

Current dataset statistics:
```
Total AHSP Items: 55
Valid Items: 46 (83.64%)
Training Samples: 274
Data Loss: 2.49% (acceptable)
```

---

## 📁 Project Structure

```
ml-rancanghub/
├── app/
│   ├── ml_v2/
│   │   ├── __init__.py
│   │   ├── data_extractor.py       # Database queries
│   │   ├── feature_builder.py      # Feature engineering
│   │   ├── trainer.py              # Model training
│   │   ├── predictor.py            # Prediction service
│   │   ├── test_data_quality.py    # Data validation
│   │   ├── test_prediction.py      # Prediction tests
│   │   ├── auto_retrain.py         # Auto-retrain orchestrator
│   │   └── notifications.py        # Email/Slack notifications
│   ├── db.py                       # Database connection
│   └── config.py                   # Configuration
├── models_v2/                      # Trained models (versioned)
├── models_v2_backups/              # Model backups
├── logs/
│   └── auto_retrain/               # Training logs
├── scripts/
│   ├── trigger_retrain.bat         # Manual retrain trigger
│   └── setup_task_simple.bat       # Task scheduler setup
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables
├── .gitignore
└── README.md
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_HOST` | Database host | `localhost` |
| `DB_PORT` | Database port | `5432` |
| `DB_USER` | Database user | - |
| `DB_PASSWORD` | Database password | - |
| `DB_NAME` | Database name | - |
| `MODEL_DIR` | Model storage directory | `./models_v2` |
| `MIN_COEFFICIENT_VALUE` | Minimum valid coefficient | `0.001` |

### Model Hyperparameters

Default hyperparameters in `trainer.py`:

```python
# Random Forest
rf_params = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 5,
    'random_state': 42
}

# XGBoost
xgb_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': 42
}
```

Adjust these in `trainer.py` for better performance.

---

## 🛠️ Troubleshooting

### Common Issues

#### Issue 1: "Invalid coefficient value for resource L-02"

**Cause**: Coefficient value too small (< 0.001)

**Solution**:
```bash
# Adjust threshold in data_extractor.py
MIN_COEFFICIENT_VALUE = 0.001  # Or lower: 0.0001

# Retrain model
python -m app.ml_v2.trainer
```

#### Issue 2: "Not enough data for training"

**Cause**: Insufficient completed projects

**Solution**:
- Add more completed projects to database
- Lower minimum requirements in `auto_retrain.py`:
  ```python
  MIN_PROJECTS_FOR_RETRAIN = 3  # Default: 5
  MIN_PROJECT_ITEMS_FOR_RETRAIN = 50  # Default: 100
  ```

#### Issue 3: Prediction returns empty results

**Cause**: Volume threshold too high

**Solution**:
```python
# Lower threshold
predictions = predict_for_new_project(
    ...,
    volume_threshold=0.0  # Or None
)
```

#### Issue 4: "Python not found" error

**Cause**: Virtual environment not activated

**Solution**:
```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

#### Issue 5: "Module not found" error

**Cause**: Missing dependencies

**Solution**:
```bash
pip install -r requirements.txt
```

### Debug Mode

Enable verbose logging:

```python
# In any ML script
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Ensure all tests pass before submitting PR

---

## 📊 Performance Benchmarks

Current model performance (as of v2-20260102):

| Metric | Random Forest | XGBoost | Ensemble |
|--------|---------------|---------|----------|
| Training MAE | 42.15 | 48.31 | 45.23 |
| Validation MAE | 50.89 | 54.45 | 52.67 |
| Training R² | 0.81 | 0.75 | 0.78 |
| Validation R² | 0.73 | 0.69 | 0.71 |

*Note: Metrics will improve as more training data becomes available*

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- RancangHub team for domain expertise
- Construction industry standards (SNI, AHSP)
- Open source ML libraries: scikit-learn, XGBoost, pandas

---

## 📧 Support

For support and questions:
- Email: support@rancanghub.com
- Issues: [GitHub Issues](https://github.com/yourusername/ml-rancanghub/issues)
- Documentation: [Wiki](https://github.com/yourusername/ml-rancanghub/wiki)

---

## 🗺️ Roadmap

### Short-term (Q1 2026)
- [ ] Implement auto-retrain scheduler
- [ ] Add model monitoring dashboard
- [ ] Improve feature engineering
- [ ] Add more unit tests

### Long-term (2026)
- [ ] Multi-model support (per project type)
- [ ] Real-time model updates
- [ ] Advanced hyperparameter tuning
- [ ] Integration with external data sources

---

**Made with ❤️ for the Indonesian construction industry**
