import { Link } from 'react-router-dom';
import { Github, ArrowRight, Youtube, Mail, MessageCircle, Twitter, Instagram, Music2 } from 'lucide-react';
import ObserverLanding from './ObserverLanding';
import DownloadsSection from './DownloadsSection';
import EnterpriseSection from './EnterpriseSection';
import RecipeBuilder from './RecipeBuilder';
import CommunityAgents from './CommunityAgents';

const LandingPage = () => {
  return (
    <div className="min-h-screen bg-[#0D1321] text-white overflow-x-hidden">
      {/* Navigation - old style with logo */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-[#0D1321]/80 backdrop-blur-xl border-b border-white/5">
        <div className="container mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-6 -ml-1">
            <img src="/eye-logo-white.svg" alt="Observer AI Logo" className="w-6 h-6" />
            <span className="text-xl font-bold font-golos hidden md:inline">Observer AI</span>
          </div>
          <div className="flex items-center space-x-7 md:space-x-9 -mr-2">
            <button onClick={() => document.getElementById('downloads')?.scrollIntoView({ behavior: 'smooth' })} className="text-gray-400 hover:text-white transition hidden md:inline">Download</button>
            <button onClick={() => document.getElementById('agents')?.scrollIntoView({ behavior: 'smooth' })} className="text-gray-400 hover:text-white transition hidden md:inline">Agents</button>
            <Link to="/howitworks" className="text-gray-400 hover:text-white transition hidden md:inline">How it works</Link>
            <a href="https://discord.gg/wnBb7ZQDUC" className="text-gray-400 hover:text-white transition hidden md:inline">Community</a>
            <a href="https://github.com/Roy3838/Observer" className="flex items-center space-x-2 bg-white/10 px-4 py-2 rounded-lg hover:bg-white/20 transition">
              <Github className="w-5 h-5" />
              <span className="hidden md:inline">GitHub</span>
            </a>
          </div>
        </div>
      </nav>

      {/* Hero Section - The Eye */}
      <header className="relative bg-[#0D1321]">
        <ObserverLanding />
      </header>

      {/* Sub-hero: build an agent in one sentence, replaces the old static "How Agents Work" diagram */}
      <RecipeBuilder />

      {/* Community Agents */}
      <CommunityAgents />

      {/* Enterprise Section */}
      <EnterpriseSection />

      {/* Downloads Section */}
      <DownloadsSection />

      {/* CTA Section */}
      <section className="py-24 bg-[#0D1321]">
        <div className="container mx-auto px-6 text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            Open source. Community driven.
          </h2>
          <p className="text-gray-400 max-w-lg mx-auto mb-10">
            Observer is built in the open. Create agents, contribute code,
            or just come hang out.
          </p>
          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <a
              href="https://app.observer-ai.com"
              className="group bg-white text-gray-900 px-8 py-4 rounded-lg font-semibold hover:bg-gray-100 transition flex items-center justify-center space-x-2"
            >
              <span>Start Building</span>
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </a>
            <a
              href="https://github.com/Roy3838/Observer"
              className="flex items-center justify-center space-x-2 bg-white/10 hover:bg-white/15 px-8 py-4 rounded-lg font-semibold transition"
            >
              <Github className="w-5 h-5" />
              <span>Star on GitHub</span>
            </a>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="pt-16 pb-8 border-t border-white/5 bg-[#0a0e17]">
        <div className="container mx-auto px-6">
          <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-10">
            {/* Brand */}
            <div className="max-w-xs">
              <div className="flex items-center space-x-3 mb-3">
                <img src="/eye-logo-white.svg" alt="Observer AI Logo" className="w-6 h-6 opacity-70" />
                <span className="text-gray-300 font-semibold">Observer AI</span>
              </div>
              <p className="text-sm text-gray-500">
                Open-source micro-agents that watch your screen and act, running entirely on your machine.
              </p>
            </div>

            {/* Link columns */}
            <div className="flex flex-wrap gap-x-16 gap-y-8">
              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-3">Follow</h4>
                <ul className="space-y-2 text-sm">
                  {[
                    { href: "https://github.com/Roy3838/Observer", label: "GitHub", Icon: Github },
                    { href: "https://discord.gg/wnBb7ZQDUC", label: "Discord", Icon: MessageCircle },
                    { href: "https://x.com/AppObserverAI", label: "X / Twitter", Icon: Twitter },
                    { href: "https://www.youtube.com/@Observer-AI", label: "YouTube", Icon: Youtube },
                    { href: "https://www.tiktok.com/@observerai", label: "TikTok", Icon: Music2 },
                    { href: "https://www.instagram.com/observer_ai.tech/", label: "Instagram", Icon: Instagram },
                  ].map(({ href, label, Icon }) => (
                    <li key={label}>
                      <a
                        href={href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-2 text-gray-400 hover:text-white transition"
                      >
                        <Icon className="w-4 h-4" />
                        <span>{label}</span>
                      </a>
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-3">Company</h4>
                <ul className="space-y-2 text-sm">
                  <li><Link to="/privacy" className="text-gray-400 hover:text-white transition">Privacy</Link></li>
                  <li><Link to="/terms" className="text-gray-400 hover:text-white transition">Terms</Link></li>
                  <li><Link to="/howitworks" className="text-gray-400 hover:text-white transition">How it works</Link></li>
                </ul>
              </div>

              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-3">Contact us</h4>
                <a
                  href="mailto:help@observer-ai.com"
                  className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition"
                >
                  <Mail className="w-4 h-4" />
                  <span>help@observer-ai.com</span>
                </a>
              </div>
            </div>
          </div>

          <div className="mt-12 pt-6 border-t border-white/5 text-xs text-gray-600">
            © {new Date().getFullYear()} Observer AI. Open source under the AGPL-3.0 license.
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
