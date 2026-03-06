import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import WindowTitleBar from "./WindowTitleBar";
import ThoughtInput from "./ThoughtInput";
import SubmitButton from "./SubmitButton";
import ResultsPanel from "./ResultsPanel";
import { useSentimentAnalysis } from "@/hooks/useSentimentAnalysis";
import { useToast } from "@/hooks/use-toast";

const SentimentWindow = () => {
  const [inputText, setInputText] = useState("");
  const { analyze, isLoading, error, results } = useSentimentAnalysis();
  const { toast } = useToast();

  const handleAnalyze = async () => {
    if (!inputText.trim()) return;
    await analyze(inputText);
  };

  // Show error toast when analysis fails
  useEffect(() => {
    if (error) {
      toast({
        title: "Analysis Failed",
        description: error,
        variant: "destructive",
      });
    }
  }, [error, toast]);

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
              disabled={isLoading}
            />
            <SubmitButton
              onClick={handleAnalyze}
              isLoading={isLoading}
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
