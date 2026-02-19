import { motion } from "framer-motion";

const MeshGradientBackground = () => {
  return (
    <div className="fixed inset-0 overflow-hidden -z-10">
      {/* Base dark background */}
      <div className="absolute inset-0 bg-background" />
      
      {/* Animated gradient orbs */}
      <motion.div
        className="absolute w-[800px] h-[800px] rounded-full opacity-30 blur-[120px]"
        style={{
          background: "radial-gradient(circle, hsl(270 60% 40%) 0%, transparent 70%)",
          left: "10%",
          top: "20%",
        }}
        animate={{
          x: [0, 100, 0],
          y: [0, 50, 0],
          scale: [1, 1.2, 1],
        }}
        transition={{
          duration: 20,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />
      
      <motion.div
        className="absolute w-[600px] h-[600px] rounded-full opacity-25 blur-[100px]"
        style={{
          background: "radial-gradient(circle, hsl(220 70% 35%) 0%, transparent 70%)",
          right: "10%",
          bottom: "20%",
        }}
        animate={{
          x: [0, -80, 0],
          y: [0, -60, 0],
          scale: [1, 1.15, 1],
        }}
        transition={{
          duration: 15,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 2,
        }}
      />
      
      <motion.div
        className="absolute w-[500px] h-[500px] rounded-full opacity-20 blur-[80px]"
        style={{
          background: "radial-gradient(circle, hsl(280 50% 35%) 0%, transparent 70%)",
          left: "50%",
          top: "50%",
          transform: "translate(-50%, -50%)",
        }}
        animate={{
          scale: [1, 1.3, 1],
          opacity: [0.2, 0.35, 0.2],
        }}
        transition={{
          duration: 12,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 1,
        }}
      />

      {/* Subtle grid overlay */}
      <div 
        className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage: `
            linear-gradient(hsl(var(--foreground)) 1px, transparent 1px),
            linear-gradient(90deg, hsl(var(--foreground)) 1px, transparent 1px)
          `,
          backgroundSize: "60px 60px",
        }}
      />
    </div>
  );
};

export default MeshGradientBackground;
