import { motion } from "framer-motion";

interface WindowTitleBarProps {
  title: string;
}

const WindowTitleBar = ({ title }: WindowTitleBarProps) => {
  return (
    <div className="flex items-center justify-between px-5 py-4 border-b border-border/50">
      {/* Title */}
      <div className="flex items-center gap-3">
        <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
        <h1 className="font-mono text-sm font-medium tracking-wider text-muted-foreground uppercase">
          {title}
        </h1>
      </div>
      
      {/* Traffic light controls */}
      <div className="flex items-center gap-2">
        <motion.button
          whileHover={{ scale: 1.15 }}
          whileTap={{ scale: 0.95 }}
          className="w-3.5 h-3.5 rounded-full bg-[#FF5F57] hover:brightness-110 transition-all shadow-sm"
          aria-label="Close"
        />
        <motion.button
          whileHover={{ scale: 1.15 }}
          whileTap={{ scale: 0.95 }}
          className="w-3.5 h-3.5 rounded-full bg-[#FEBC2E] hover:brightness-110 transition-all shadow-sm"
          aria-label="Minimize"
        />
        <motion.button
          whileHover={{ scale: 1.15 }}
          whileTap={{ scale: 0.95 }}
          className="w-3.5 h-3.5 rounded-full bg-[#28C840] hover:brightness-110 transition-all shadow-sm"
          aria-label="Maximize"
        />
      </div>
    </div>
  );
};

export default WindowTitleBar;
