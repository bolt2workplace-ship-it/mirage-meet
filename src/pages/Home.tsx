import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Logo from '../components/Logo';
import { Video, ArrowRight, Copy, Check, Link2 } from 'lucide-react';

export default function Home() {
  const navigate = useNavigate();
  const [displayName, setDisplayName] = useState('');
  const [roomCode, setRoomCode] = useState('');
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState('');
  const [pendingRoomId, setPendingRoomId] = useState<string | null>(null);

  const handleCreateRoom = async () => {
    setError('');
    try {
      const response = await fetch('/create-room', {
        method: 'POST',
      });
      const data = await response.json();
      setPendingRoomId(data.roomId);
    } catch (err) {
      setError('Failed to create room. Please try again.');
    }
  };

  const handleJoinRoom = async () => {
    if (!roomCode.trim()) {
      setError('Please enter a room code');
      return;
    }
    navigate(`/meeting/${roomCode.trim()}?join=true`);
  };

  return (
    <div className="min-h-screen gradient-dark">
      <header className="absolute top-0 left-0 right-0 p-4 md:p-6 z-10">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <Logo size="md" />
        </div>
      </header>

      <div className="min-h-screen flex flex-col items-center justify-center px-4">
        <div className="max-w-2xl w-full text-center">
          <div className="mb-8">
            <div className="inline-flex items-center justify-center p-3 bg-gradient-to-br from-primary-400 to-primary-600 rounded-2xl shadow-lg shadow-primary-600/20 mb-6">
              <Video size={40} className="text-white" />
            </div>
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-4">
              Mirage Meet
            </h1>
            <p className="text-lg md:text-xl text-dark-300 max-w-lg mx-auto">
              Professional video meetings with advanced features. Connect instantly with anyone, anywhere.
            </p>
          </div>

          <div className="flex flex-col-reverse sm:flex-col gap-6">
            <div className="bg-dark-800/50 backdrop-blur-sm border border-dark-700 rounded-2xl p-6 md:p-8">
              <h2 className="text-lg font-semibold text-white mb-4">Join a Meeting</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-dark-300 text-left mb-2">
                    Display Name
                  </label>
                  <input
                    type="text"
                    value={displayName}
                    onChange={(e) => setDisplayName(e.target.value)}
                    placeholder="Enter your name"
                    className="w-full px-4 py-3 bg-dark-900 border border-dark-600 rounded-lg text-white placeholder-dark-500 focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500 transition-all"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-dark-300 text-left mb-2">
                    Room Code
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={roomCode}
                      onChange={(e) => setRoomCode(e.target.value.toUpperCase())}
                      placeholder="e.g. ABC-123-XYZ"
                      className="flex-1 px-4 py-3 bg-dark-900 border border-dark-600 rounded-lg text-white placeholder-dark-500 focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500 transition-all uppercase tracking-wider"
                    />
                    <button
                      onClick={handleJoinRoom}
                      className="px-6 py-3 bg-primary-500 text-white rounded-lg font-medium hover:bg-primary-600 transition-colors flex items-center gap-2"
                    >
                      Join
                      <ArrowRight size={18} />
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row gap-4">
              <button
                onClick={handleCreateRoom}
                className="flex-1 inline-flex items-center justify-center gap-3 px-6 py-4 bg-gradient-to-r from-primary-500 to-primary-600 text-white rounded-xl font-semibold text-lg hover:from-primary-600 hover:to-primary-700 transition-all shadow-lg shadow-primary-500/25"
              >
                <Video size={22} />
                Create New Meeting
              </button>
            </div>

            {pendingRoomId && (
              <div className="bg-dark-800/50 backdrop-blur-sm border border-dark-700 rounded-2xl p-6 mt-6">
                <h3 className="text-lg font-semibold text-white mb-2">Room Created!</h3>
                <p className="text-sm text-dark-300 mb-4">Share this link with your participants:</p>
                <div className="flex items-center gap-2">
                  <div className="flex-1 px-4 py-3 bg-dark-900 border border-dark-600 rounded-lg text-primary-400 font-mono text-sm overflow-x-auto">
                    {`${window.location.origin}/meeting/${pendingRoomId}`}
                  </div>
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText(`${window.location.origin}/meeting/${pendingRoomId}`);
                      setCopied(true);
                      setTimeout(() => setCopied(false), 2000);
                    }}
                    className="px-4 py-3 bg-dark-700 hover:bg-dark-600 text-white rounded-lg transition-colors flex items-center gap-2"
                  >
                    {copied ? <Check size={18} /> : <Copy size={18} />}
                    {copied ? 'Copied!' : 'Copy'}
                  </button>
                </div>
                <button
                  onClick={() => navigate(`/meeting/${pendingRoomId}?host=true`)}
                  className="w-full mt-4 px-6 py-3 bg-primary-500 hover:bg-primary-600 text-white rounded-lg font-medium transition-colors"
                >
                  Start Meeting
                </button>
              </div>
            )}
          </div>

          {error && (
            <div className="mt-4 px-4 py-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
              {error}
            </div>
          )}

          <div className="mt-12 flex flex-wrap items-center justify-center gap-8 text-dark-500 text-sm">
            <div className="flex items-center gap-2">
              <Link2 size={16} />
              <span>End-to-end encrypted</span>
            </div>
            <div className="flex items-center gap-2">
              <Video size={16} />
              <span>HD Video</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
