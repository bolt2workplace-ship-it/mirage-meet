import { Video } from 'lucide-react';

export default function Logo({ size = 'md' }: { size?: 'sm' | 'md' | 'lg' }) {
  const sizes = {
    sm: { icon: 20, text: 'text-lg', gap: 'gap-1.5' },
    md: { icon: 28, text: 'text-2xl', gap: 'gap-2' },
    lg: { icon: 36, text: 'text-3xl', gap: 'gap-2.5' },
  };

  const s = sizes[size];

  return (
    <div className={`flex items-center ${s.gap}`}>
      <div className="relative">
        <div className="absolute inset-0 bg-gradient-to-br from-primary-400 to-primary-600 rounded-lg blur opacity-50" />
        <div className="relative bg-gradient-to-br from-primary-400 to-primary-600 rounded-lg p-1.5">
          <Video size={s.icon} className="text-white" />
        </div>
      </div>
      <div className="flex flex-col leading-none">
        <span className={`font-bold ${s.text} text-white tracking-tight`}>
          Mirage
        </span>
        <span className={`font-medium ${s.text} text-primary-400 tracking-wide -mt-0.5`}>
          Meet
        </span>
      </div>
    </div>
  );
}
