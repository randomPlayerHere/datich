import { useState } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface SentimentMetric {
  label: string;
  value: number;
}

export interface ProfileMatch {
  profile: string;
  confidence_percentage: number;
}

export interface Classification {
  primary_profile: string;
  top_3_matches: ProfileMatch[];
}

export interface SentimentResults {
  metrics: SentimentMetric[];
  classification: Classification;
}

interface ApiScores {
  sadness: number;
  anxiety: number;
  rumination: number;
  self_focus: number;
  hopelessness: number;
  emotional_volatility: number;
}

interface ApiResponse {
  success: boolean;
  data: {
    scores: ApiScores;
    classification: Classification;
  } | null;
  message: string;
  model_version: string;
}

const SCORE_LABELS: Record<keyof ApiScores, string> = {
  sadness: 'Sadness',
  anxiety: 'Anxiety',
  rumination: 'Rumination',
  self_focus: 'Self Focus',
  hopelessness: 'Hopelessness',
  emotional_volatility: 'Emotional Volatility',
};

export function useSentimentAnalysis() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<SentimentResults | null>(null);

  const analyze = async (text: string): Promise<SentimentResults | null> => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: ApiResponse = await response.json();

      if (!data.success || !data.data) {
        throw new Error(data.message || 'Analysis failed');
      }

      // Map API scores (0-1 floats) to percentage metrics (0-100 integers)
      const metrics: SentimentMetric[] = (Object.keys(SCORE_LABELS) as (keyof ApiScores)[]).map(
        (key) => ({
          label: SCORE_LABELS[key],
          value: Math.round(data.data!.scores[key] * 100),
        })
      );

      const mapped: SentimentResults = {
        metrics,
        classification: data.data.classification,
      };

      setResults(mapped);
      return mapped;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'An error occurred';
      setError(message);
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  const reset = () => {
    setResults(null);
    setError(null);
  };

  return { analyze, isLoading, error, results, setResults, reset };
}
