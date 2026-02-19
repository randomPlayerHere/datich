import { useState } from "react";
import { motion } from "framer-motion";
import confetti from "canvas-confetti";
import WindowTitleBar from "./WindowTitleBar";
import ThoughtInput from "./ThoughtInput";
import SubmitButton from "./SubmitButton";
import ResultsPanel, { SentimentResults } from "./ResultsPanel";

const SentimentWindow = () => {
  const [inputText, setInputText] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<SentimentResults | null>(null);

  const simulateAnalysis = async () => {
    if (!inputText.trim()) return;
    
    setIsAnalyzing(true);
    setResults(null);
    
    // Simulate 2-second analysis delay
    await new Promise((resolve) => setTimeout(resolve, 2000));
    
    // Generate random results with dynamic labels
    const newResults: SentimentResults = {
      metrics: [
        { label: "Anxiety Indicators", value: Math.floor(Math.random() * 100) },
        { label: "Mood Stability", value: Math.floor(Math.random() * 100) },
        { label: "Stress Level", value: Math.floor(Math.random() * 100) },
      ]
    };
    
    setResults(newResults);
    setIsAnalyzing(false);
    
    // Trigger confetti if mood is high
    const moodMetric = newResults.metrics.find(m => m.label.toLowerCase().includes('mood'));
    if (moodMetric && moodMetric.value >= 70) {
      triggerSparkles();
    }
  };

  const triggerSparkles = () => {
    confetti({
      particleCount: 30,
      spread: 60,
      origin: { x: 0.75, y: 0.4 },
      colors: ['#28C840', '#4ade80', '#22c55e'],
      gravity: 1.5,
      scalar: 0.8,
      ticks: 100,
    });
  };

  return (
    <motion.div
      initial={{ scale: 0.9, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{
        type: "spring",
        stiffness: 300,
        damping: 30,
      }}
      className="glass-window rounded-2xl w-full max-w-4xl mx-4 overflow-hidden"
    >
      {/* Title Bar */}
      <WindowTitleBar title="Mental Health Sentiment OS v1.0" />
      
      {/* Content */}
      <div className="p-6 md:p-8">
        <div className="grid md:grid-cols-2 gap-6 md:gap-8">
          {/* Left Column - Input */}
          <div className="flex flex-col gap-5">
            <ThoughtInput
              value={inputText}
              onChange={setInputText}
              disabled={isAnalyzing}
            />
            <SubmitButton
              onClick={simulateAnalysis}
              isLoading={isAnalyzing}
              disabled={!inputText.trim()}
            />
          </div>
          
          {/* Right Column - Results */}
          <div className="min-h-[300px]">
            <ResultsPanel
              results={results}
              isVisible={!!results}
            />
          </div>
        </div>
      </div>
      
      {/* Footer Disclaimer */}
      <div className="px-6 py-4 border-t border-border/30 bg-muted/10">
        <p className="font-mono text-[10px] text-muted-foreground/60 text-center tracking-wide">
          Project Purpose: Portfolio demonstration of SLM integration. Not for medical use.
        </p>
      </div>
    </motion.div>
  );
};

export default SentimentWindow;
