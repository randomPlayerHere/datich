import { motion } from "framer-motion";
import SentimentBar from "./SentimentBar";

export interface SentimentResult {
  label: string;
  value: number;
}

export interface SentimentResults {
  metrics: SentimentResult[];
}

interface ResultsPanelProps {
  results: SentimentResults | null;
  isVisible: boolean;
}

const ResultsPanel = ({ results, isVisible }: ResultsPanelProps) => {
  if (!isVisible || !results) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="flex flex-col justify-center items-center h-full text-center p-6"
      >
        <div className="w-16 h-16 rounded-2xl bg-muted/30 flex items-center justify-center mb-4">
          <motion.div
            animate={{ 
              rotate: [0, 10, -10, 0],
              scale: [1, 1.05, 1],
            }}
            transition={{ 
              duration: 4, 
              repeat: Infinity,
              ease: "easeInOut",
            }}
            className="text-3xl"
          >
            🧠
          </motion.div>
        </div>
        <p className="font-mono text-xs text-muted-foreground tracking-wide uppercase mb-2">
          Awaiting Input
        </p>
        <p className="text-sm text-muted-foreground/70 max-w-[200px]">
          Enter your thoughts and submit to begin analysis
        </p>
      </motion.div>
    );
  }

  const codes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="flex flex-col h-full"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-6">
        <div className="w-2 h-2 rounded-full bg-indicator-low animate-pulse" />
        <span className="font-mono text-xs text-muted-foreground tracking-wider uppercase">
          Analysis Complete
        </span>
      </div>
      
      {/* Results */}
      <div className="flex flex-col gap-6 flex-1">
        {results.metrics.map((metric, index) => (
          <SentimentBar
            key={index}
            code={codes[index] || String(index + 1)}
            label={metric.label}
            value={metric.value}
            delay={index * 0.15}
            showSparkle={metric.label.toLowerCase().includes('mood') && metric.value >= 70}
          />
        ))}
      </div>
    </motion.div>
  );
};

export default ResultsPanel;
