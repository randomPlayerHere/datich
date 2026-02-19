# Datich

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688?logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)
![React](https://img.shields.io/badge/React-18%2B-61DAFB?logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0%2B-3178C6?logo=typescript)
![Transformers](https://img.shields.io/badge/Transformers-NLP-FFD21E?logo=huggingface)

A sentiment analysis platform that combines machine learning with a modern web interface. Datich analyzes text input and provides dynamic sentiment metrics with visual feedback.

**Status**: This project is currently under active development. Core features are functional, with ongoing improvements to the ML model, backend API, and UI/UX.

## Overview

Datich consists of three main components:

- **Frontend**: A React + TypeScript web application with a polished UI using Tailwind CSS and shadcn/ui components
- **Backend**: A FastAPI server providing sentiment analysis API endpoints
- **ML**: Machine learning models and data processing pipeline for training custom sentiment classifiers

The application allows users to input text and receive detailed sentiment analysis results with visual representations of emotional metrics.

## Project Structure

```
datich/
├── frontend/              # React + Vite frontend application
│   ├── src/
│   │   ├── components/   # Reusable UI components
│   │   ├── pages/        # Page components
│   │   ├── hooks/        # Custom React hooks
│   │   └── lib/          # Utilities and helpers
│   └── package.json
├── backend/              # FastAPI backend server
│   ├── models/           # Sentiment analysis models and schemas
│   ├── routers/          # API route handlers
│   ├── config.py         # Configuration
│   └── main.py           # Application entry point
├── ml/                   # Machine learning pipeline
│   ├── training.ipynb    # Model training notebook
│   ├── mpv.ipynb         # Data exploration and processing
│   ├── data/             # Datasets (raw and processed)
│   └── utils/            # ML utility functions
└── db/                   # Database files (when implemented)
```

## Tech Stack

### Frontend
- React 18 with TypeScript
- Vite for fast build tooling
- Tailwind CSS for styling
- shadcn/ui component library
- React Query for data fetching
- React Router for navigation

### Backend
- FastAPI for REST API
- Transformers (Hugging Face) for NLP models
- PyTorch for deep learning
- Sentence Transformers for embeddings
- Uvicorn ASGI server

### Machine Learning
- PyTorch for model training
- Sentence Transformers
- Pandas for data manipulation
- Scikit-learn for preprocessing

## Getting Started

### Prerequisites

- Node.js 18+ and npm (for frontend)
- Python 3.9+ (for backend and ML)
- Git

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`

### Backend Setup

Install PyTorch (CPU version):

```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
```

Install backend dependencies:

```bash
cd backend
pip install -r requirements.txt
```

Run the backend server:

```bash
python main.py
```

The API will be available at `http://localhost:8000` with interactive docs at `/docs`

### Machine Learning Setup

```bash
pip install -r ml/requirements.txt
jupyter notebook
```

Open `ml/training.ipynb` or `ml/mpv.ipynb` to work with the ML pipeline.

## Available Scripts

### Frontend

```bash
npm run dev              # Start development server
npm run build            # Build for production
npm run preview          # Preview production build
npm run lint             # Run ESLint
npm run test             # Run tests
npm run test:watch       # Run tests in watch mode
```

### Backend

```bash
python main.py           # Start FastAPI server
```

## API Endpoints

The backend provides the following endpoints:

- `GET /` - Health check
- `GET /health` - API health status
- `POST /api/v1/analyze` - Analyze sentiment of provided text (in development)

API documentation available at `http://localhost:8000/docs`

## Features

### Current
- Clean, modern web interface with mesh gradient background
- Text input for sentiment analysis
- Responsive design that works on desktop and mobile
- Backend API structure ready for integration
- ML model pipeline with data processing

### In Development
- Full sentiment analysis integration
- Dynamic sentiment metrics visualization
- Support for multiple sentiment categories (positive, negative, neutral, etc.)
- Advanced ML model training with custom datasets
- Results caching and history tracking

## Development Workflow

1. **Frontend Development**: Make changes in `frontend/src/` and the dev server will hot reload
2. **Backend Development**: Modify files in `backend/` and restart the server
3. **ML Development**: Work in Jupyter notebooks in `ml/` folder for experimentation

## Deployment

The frontend can be deployed to Vercel, Netlify, or any static hosting service:

```bash
cd frontend
npm run build
# Upload the dist/ folder to your hosting provider
```

### Vercel Deployment (Recommended)

1. Push your code to GitHub
2. Connect your repository to Vercel
3. Set root directory to `frontend`
4. Click deploy

The backend can be deployed to services like Heroku, Railway, or AWS EC2.

## Environment Variables

### Backend
Create a `.env` file in the `backend/` directory:

```
MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
CUDA_AVAILABLE=false
```

### Frontend
The frontend uses environment variables for API endpoints (update as needed for production).

## Known Limitations

- Backend sentiment analysis endpoints are not yet integrated with the frontend
- ML model training is still in experimental phase
- Database integration is pending
- No user authentication system currently implemented

## Future Roadmap

- Complete backend-frontend integration
- Deploy trained ML model to production
- Add user accounts and history tracking
- Implement result persistence with database
- Add support for multiple languages
- Real-time sentiment analysis streaming
- Export analysis results

## Contributing

This is a solo project under active development. Contributions and suggestions are welcome.

## Project Notes

This project was built with a focus on clean architecture and modern web development practices. The ML pipeline was trained on real-world sentiment data sourced from Reddit discussions on anxiety, depression, loneliness, mental health, and social issues.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions about this project, feel free to open an issue or discussion.
