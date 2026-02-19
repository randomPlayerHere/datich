import { useState } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface SentimentMetric {
  label: string;
  value: number;
}

export interface SentimentResults {
  metrics: SentimentMetric[];
}

interface AnalysisResponse {
  success: boolean;
  data: SentimentResults | null;
  message: string;
  model_version: string;
}

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

      const data: AnalysisResponse = await response.json();

      if (!data.success || !data.data) {
        throw new Error(data.message || 'Analysis failed');
      }

      setResults(data.data);
      return data.data;
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
