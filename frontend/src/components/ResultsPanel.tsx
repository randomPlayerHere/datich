import { motion } from "framer-motion";
import SentimentBar from "./SentimentBar";

export interface ProfileMatch {
  profile: string;
  confidence_percentage: number;
}

export interface Classification {
  primary_profile: string;
  top_3_matches: ProfileMatch[];
}

export interface SentimentResult {
  label: string;
  value: number;
}

export interface SentimentResults {
  metrics: SentimentResult[];
  classification: Classification;
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

  const codes = ['A', 'B', 'C', 'D', 'E', 'F'];

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

      {/* Classification Results */}
      {results.classification && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="flex flex-col justify-center flex-1"
        >
          {/* Primary Profile */}
          <div className="flex items-center gap-2 mb-2">
            <div className="w-2 h-2 rounded-full bg-primary" />
            <span className="font-mono text-[10px] text-muted-foreground tracking-wider uppercase">
              Primary Profile
            </span>
          </div>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="font-mono text-sm font-semibold text-foreground mb-6"
          >
            {results.classification.primary_profile}
          </motion.p>

          {/* Top 3 Matches as Bars */}
          <div className="flex flex-col gap-6">
            {results.classification.top_3_matches.map((match, i) => (
              <SentimentBar
                key={i}
                code={codes[i]}
                label={match.profile}
                value={Math.round(match.confidence_percentage)}
                delay={i * 0.15}
              />
            ))}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};

export default ResultsPanel;
