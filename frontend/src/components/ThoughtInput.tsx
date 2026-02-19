import { motion } from "framer-motion";

interface ThoughtInputProps {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}

const ThoughtInput = ({ value, onChange, disabled }: ThoughtInputProps) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2, duration: 0.5 }}
      className="flex flex-col h-full"
    >
      <label className="font-mono text-xs font-medium text-muted-foreground mb-3 tracking-wider uppercase flex items-center gap-2">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-secondary" />
        Input Stream
      </label>
      
      <div className="relative flex-1">
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          placeholder="Enter your thoughts..."
          className="w-full h-full min-h-[180px] p-4 rounded-xl bg-input/50 border border-border/50 
                     text-foreground placeholder:text-muted-foreground/50 
                     resize-none font-sans text-sm leading-relaxed
                     input-glow focus:outline-none focus:border-primary/50
                     disabled:opacity-50 disabled:cursor-not-allowed
                     transition-all duration-300"
        />
        
        {/* Decorative corner accent */}
        <div className="absolute -bottom-1 -right-1 w-8 h-8 border-r-2 border-b-2 border-primary/30 rounded-br-xl pointer-events-none" />
      </div>
    </motion.div>
  );
};

export default ThoughtInput;
