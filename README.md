<div align="center">

# Datich

**Mental state analysis powered by a fine-tuned Small Language Model**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8-3178C6?style=flat&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/HF%20Spaces-Live-FF9D00?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/randomPlayerHere/datich-hf)

---

Datich analyzes text for mental health indicators using a **Qwen2.5-0.5B** model fine-tuned with LoRA adapters. It scores six emotional dimensions and classifies the input into one of four psychological profiles.

> **Disclaimer**: This is a portfolio project demonstrating SLM integration. It is not intended for clinical or medical use.

</div>

---

## Architecture

```
Frontend (React/Vite)          Backend (FastAPI on HF Spaces)
---------------------          --------------------------------
                               Qwen2.5-0.5B + LoRA adapter
  User Input ----POST----->    Tokenize -> Forward Pass -> 6 scores
                               StandardScaler -> KMeans clustering
  Profile Bars <--JSON-----    Classification (4 profiles)
```

The frontend sends text to the FastAPI backend hosted on Hugging Face Spaces. The backend runs a single forward pass through the fine-tuned model, producing six emotion scores. These scores are then scaled and clustered via a pre-trained KMeans model to produce a psychological profile classification.

---

## Emotional Dimensions

| Score | Description |
|-------|-------------|
| Sadness | Degree of sadness indicators in the text |
| Anxiety | Level of anxious or worried language |
| Rumination | Repetitive, self-reflective thought patterns |
| Self Focus | Degree of inward-directed attention |
| Hopelessness | Indicators of despair or lack of future orientation |
| Emotional Volatility | Instability or rapid shifts in emotional tone |

## Profile Classifications

| Profile | Description |
|---------|-------------|
| Severe Distress / Depressive Profile | High scores across most negative dimensions |
| Passive Sadness / Apathy | Elevated sadness with low volatility |
| Baseline / Mild Anxiety | Low overall scores, minor anxious tendencies |
| Emotionally Volatile / Dysregulated | High emotional volatility and instability |

---

## Project Structure

```
datich/
├── main.py                  # FastAPI application with all endpoints
├── Dockerfile               # HF Spaces Docker deployment
├── requirements.txt         # Python dependencies
├── ml/
│   ├── binaries/
│   │   ├── qwen_lora/       # Fine-tuned LoRA adapter weights
│   │   ├── datich_scaler.pkl
│   │   └── datich_kmeans_model.pkl
│   ├── model_training.py    # Training script
│   ├── data_preprocessing.py
│   └── utils/
│       └── data_labelling.py
└── frontend/
    ├── src/
    │   ├── components/      # React UI components
    │   ├── hooks/            # useSentimentAnalysis API hook
    │   └── pages/
    ├── package.json
    └── vite.config.ts
```

---

## API

The backend exposes a REST API at `https://randomplayerhere-datich-hf.hf.space`.

### `POST /api/v1/analyze`

Analyze text for mental state indicators.

**Request:**

```json
{
  "text": "I feel very anxious and worried about my future"
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "scores": {
      "sadness": 0.71,
      "anxiety": 0.57,
      "rumination": 0.60,
      "self_focus": 0.68,
      "hopelessness": 0.18,
      "emotional_volatility": 0.00
    },
    "classification": {
      "primary_profile": "Severe Distress / Depressive Profile",
      "top_3_matches": [
        { "profile": "Severe Distress / Depressive Profile", "confidence_percentage": 29.0 },
        { "profile": "Baseline / Mild Anxiety", "confidence_percentage": 27.9 },
        { "profile": "Passive Sadness / Apathy", "confidence_percentage": 25.4 }
      ]
    }
  },
  "message": "Analysis completed successfully",
  "model_version": "1.0.0"
}
```

### `GET /health`

Returns `{"status": "ok", "model_loaded": true}` when the service is ready.

### `GET /api/v1/models/info`

Returns model metadata (name, adapter, device, status).

---

## Local Development

### Backend

```bash
# Clone the repository
git clone https://github.com/randomPlayerHere/datich.git
cd datich

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API docs will be available at `http://localhost:8000/docs`.

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Create .env file
echo "VITE_API_URL=http://localhost:8000" > .env

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:8080`.

---

## Deployment

| Component | Platform | URL |
|-----------|----------|-----|
| Backend API | Hugging Face Spaces (Docker) | [randomPlayerHere/datich-hf](https://huggingface.co/spaces/randomPlayerHere/datich-hf) |
| Frontend | Vercel | -- |

When deploying the frontend to Vercel, set the environment variable:

```
VITE_API_URL=https://randomplayerhere-datich-hf.hf.space
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Model | Qwen2.5-0.5B with LoRA (PEFT) |
| Backend | FastAPI, PyTorch, scikit-learn, Transformers |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS, Framer Motion |
| Deployment | Docker on HF Spaces, Vercel |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
