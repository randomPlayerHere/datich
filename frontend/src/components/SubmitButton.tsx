import { motion, AnimatePresence } from "framer-motion";

interface SubmitButtonProps {
  onClick: () => void;
  isLoading: boolean;
  disabled?: boolean;
}

const SubmitButton = ({ onClick, isLoading, disabled }: SubmitButtonProps) => {
  return (
    <motion.button
      onClick={onClick}
      disabled={disabled || isLoading}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className="btn-submit w-full py-4 px-8 rounded-xl text-primary-foreground font-mono
                 uppercase tracking-[0.2em] text-sm relative overflow-hidden
                 disabled:opacity-70"
    >
      <AnimatePresence mode="wait">
        {isLoading ? (
          <motion.div
            key="loading"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="flex items-center justify-center gap-3"
          >
            {/* Animated spinner */}
            <svg 
              className="w-5 h-5 animate-spin-slow" 
              viewBox="0 0 24 24" 
              fill="none"
            >
              <circle 
                className="opacity-25" 
                cx="12" 
                cy="12" 
                r="10" 
                stroke="currentColor" 
                strokeWidth="3"
              />
              <path 
                className="opacity-75" 
                fill="currentColor" 
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
            <span>Analyzing...</span>
            
            {/* Pulsing dots */}
            <div className="flex gap-1">
              {[0, 1, 2].map((i) => (
                <motion.span
                  key={i}
                  className="w-1.5 h-1.5 rounded-full bg-primary-foreground"
                  animate={{
                    opacity: [0.3, 1, 0.3],
                    scale: [0.8, 1, 0.8],
                  }}
                  transition={{
                    duration: 1,
                    repeat: Infinity,
                    delay: i * 0.2,
                  }}
                />
              ))}
            </div>
          </motion.div>
        ) : (
          <motion.span
            key="submit"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            Submit
          </motion.span>
        )}
      </AnimatePresence>
      
      {/* Shimmer effect */}
      <motion.div
        className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/20 to-transparent"
        animate={{ x: ["0%", "200%"] }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: "linear",
          repeatDelay: 3,
        }}
      />
    </motion.button>
  );
};

export default SubmitButton;
