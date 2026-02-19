import { motion } from "framer-motion";
import { useState } from "react";
import { Info } from "lucide-react";

interface SentimentBarProps {
  label: string;
  code: string;
  value: number;
  delay?: number;
  showSparkle?: boolean;
}

const SentimentBar = ({ label, code, value, delay = 0, showSparkle = false }: SentimentBarProps) => {
  const [showTooltip, setShowTooltip] = useState(false);
  
  // Determine level based on value
  const getLevel = (val: number): 'low' | 'medium' | 'high' => {
    if (val <= 33) return 'low';
    if (val <= 66) return 'medium';
    return 'high';
  };
  
  const level = getLevel(value);
  
  const fillClass = {
    low: 'progress-fill-low',
    medium: 'progress-fill-medium',
    high: 'progress-fill-high',
  }[level];

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: delay + 0.3, duration: 0.5 }}
      className={`relative ${showSparkle ? 'sparkle-container active' : ''}`}
    >
      {/* Header row */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <span className="font-mono text-lg font-bold text-primary">{code}</span>
          <span className="font-mono text-xs text-muted-foreground tracking-wide uppercase">{label}</span>
        </div>
        
        {/* Value and info icon */}
        <div className="flex items-center gap-2">
          <motion.span
            key={value}
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: delay + 0.6, type: "spring", stiffness: 200 }}
            className="font-mono text-sm font-semibold text-foreground"
          >
            {value}%
          </motion.span>
          
          {/* Info icon with tooltip */}
          <div className="relative">
            <motion.button
              onMouseEnter={() => setShowTooltip(true)}
              onMouseLeave={() => setShowTooltip(false)}
              whileHover={{ scale: 1.1 }}
              className="w-5 h-5 rounded-full bg-muted/50 flex items-center justify-center
                         hover:bg-muted transition-colors"
            >
              <Info className="w-3 h-3 text-muted-foreground" />
            </motion.button>
            
            {/* Tooltip */}
            {showTooltip && (
              <motion.div
                initial={{ opacity: 0, y: 5, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 5, scale: 0.95 }}
                className="absolute right-0 top-full mt-2 z-50 w-64 p-3 rounded-lg
                           bg-card border border-border shadow-xl"
              >
                <p className="text-xs text-muted-foreground leading-relaxed">
                  This is a sentiment visualization based on text patterns, not a clinical diagnosis.
                </p>
                <div className="absolute -top-1 right-3 w-2 h-2 bg-card border-l border-t border-border rotate-45" />
              </motion.div>
            )}
          </div>
        </div>
      </div>
      
      {/* Progress bar */}
      <div className="progress-track h-3 rounded-full">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          transition={{
            delay: delay + 0.4,
            duration: 0.8,
            type: "spring",
            stiffness: 50,
            damping: 15,
          }}
          className={`h-full rounded-full ${fillClass}`}
        />
      </div>
      
      {/* Sparkle particles for high mood stability */}
      {showSparkle && (
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          {[...Array(5)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 rounded-full bg-indicator-low"
              initial={{ 
                x: `${20 + i * 15}%`, 
                y: "50%",
                opacity: 0,
                scale: 0,
              }}
              animate={{
                y: ["50%", "-20%"],
                opacity: [0, 1, 0],
                scale: [0, 1.5, 0],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                delay: i * 0.3,
                ease: "easeOut",
              }}
            />
          ))}
        </div>
      )}
    </motion.div>
  );
};

export default SentimentBar;
