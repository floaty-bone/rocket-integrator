import React, { useState, useEffect } from 'react';
import { Download, Linkedin, ChevronRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import RippleMesh from '../components/RippleMesh';

const DownloadsPage = () => {
  const [scrollPosition, setScrollPosition] = useState(0);
  const [navVisible, setNavVisible] = useState(true);

  useEffect(() => {
    let lastScroll = 0;

    const handleScroll = () => {
      const currentScroll = window.pageYOffset;
      setScrollPosition(currentScroll);
      setNavVisible(currentScroll < lastScroll || currentScroll < 50);
      lastScroll = currentScroll;
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleContactClick = (e: React.MouseEvent) => {
    e.preventDefault();

    // Scroll to bottom after short delay
    setTimeout(() => {
      window.scrollTo({
        top: document.documentElement.scrollHeight,
        behavior: 'smooth'
      });
    }, 100);
  };

  return (
    <div className="min-h-screen text-white" style={{ background: 'transparent' }}>
      {/* Full-page fixed 3D background */}
      <div className="fixed inset-0 z-0">
        <RippleMesh className="w-full h-full" />
      </div>

      {/* Navigation */}
      <nav className={`fixed w-full z-50 transition-all duration-500 ${scrollPosition > 50
        ? 'bg-black/90 backdrop-blur-sm'
        : 'bg-transparent'
        } ${navVisible
          ? 'transform translate-y-0'
          : 'transform -translate-y-full'
        }`}
      >
        <div className="w-full px-24 py-6 flex justify-between items-center">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-light tracking-wide">Ali Abouelazz</h1>
          </div>
          <div className="flex gap-12 text-sm font-light tracking-wider">
            <Link to="/home" className="hover:text-[#9F8E6D] transition-colors duration-300">HOME</Link>
            <Link to="/downloadsPage" className="hover:text-[#9F8E6D] transition-colors duration-300">TECHNICAL PORTFOLIO</Link>
            <Link to="/competencesPage" className="hover:text-[#9F8E6D] transition-colors duration-300">SKILLS</Link>
            <Link to="/loisirs" className="hover:text-[#9F8E6D] transition-colors duration-300">INTERESTS</Link>
            <a href='#' onClick={handleContactClick} className="hover:text-[#9F8E6D] transition-colors duration-300">CONTACT</a>
          </div>
        </div>
      </nav>

      {/* Downloads Content */}
      <div className="relative z-10 min-h-screen flex items-center justify-center px-24 pt-15">
        <div className="relative h-full flex items-center justify-center px-24 pt-15">
          <div className="w-full max-w-4xl">
            <h2 className="text-4xl font-light mb-12 text-white mt-16">Downloads</h2>
            <div className="flex flex-wrap justify-center gap-8">
              {/* Portfolio Technique Download Card */}
              <div className="bg-white/10 border border-white/20 rounded-lg p-8 flex flex-col items-center transition-all duration-300 hover:bg-white/15 hover:border-[#9F8E6D]/40 hover:shadow-lg hover:shadow-[#9F8E6D]/20 hover:scale-105 w-72">
                <h3 className="text-2xl mb-6">Technical Portfolio</h3>
                <p className="text-gray-300 mb-8 text-center">
                  Explore my technical projects and detailed achievements
                </p>
                <a
                  href="/ali-portfolio/downloads/Portfolio-technique.pdf"
                  download
                  className="flex items-center gap-2 bg-[#9F8E6D] hover:bg-[#7A6D54] px-8 py-3 text-white rounded-md cursor-pointer transition-colors duration-200"
                >
                  <Download className="w-5 h-5" />
                  <span>Download Portfolio</span>
                </a>
              </div>

              {/* Rocket Demo Card */}
              <div className="bg-white/10 border border-white/20 rounded-lg p-8 flex flex-col items-center transition-all duration-300 hover:bg-white/15 hover:border-[#9F8E6D]/40 hover:shadow-lg hover:shadow-[#9F8E6D]/20 hover:scale-105 w-72">
                <h3 className="text-2xl mb-6">Starship Booster Landing</h3>
                <p className="text-gray-300 mb-8 text-center">
                  Experimenting with LQR full-state feedback control — live 3D simulation
                </p>
                <Link
                  to="/rocketDemo"
                  className="flex items-center gap-2 bg-[#9F8E6D] hover:bg-[#7A6D54] px-8 py-3 text-white rounded-md cursor-pointer transition-colors duration-200"
                >
                  <ChevronRight className="w-5 h-5" />
                  <span>Launch Demo</span>
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Contact Section - Matching other pages */}
      <section className="relative z-10 py-40 px-24 mt-20">
        <div className="max-w-7xl mx-auto flex justify-between items-start" style={{ top: "10px" }}>
          <div className="max-w-lg">
            <h3 className="text-4xl font-light mb-8">Contact Me</h3>
            <div className="space-y-8">
              <div>
                <span className="text-xs font-semibold text-[#9F8E6D] uppercase tracking-widest mb-2 block">Email</span>
                <a
                  href="mailto:ali.abouelazz@gmail.com"
                  className="text-lg font-light text-white hover:text-[#9F8E6D] transition-colors duration-300"
                >
                  ali.abouelazz@gmail.com
                </a>
              </div>
              <div>
                <span className="text-xs font-semibold text-[#9F8E6D] uppercase tracking-widest mb-2 block">WhatsApp contact only</span>
                <a
                  href="tel:+33777451629"
                  className="text-lg font-light text-white hover:text-[#9F8E6D] transition-colors duration-300"
                >
                  +33 7 77 45 16 29
                </a>
              </div>
            </div>
          </div>
          <div className="w-[200px] flex justify-center items-center">
            <img
              src="/ali-portfolio/images-videos/profilePic.png"
              alt="Mohamed Ali Abouelazz Profile"
              className="w-full h-auto object-cover rounded-lg shadow-lg"
            />
          </div>
        </div>

        {/* Footer */}
        <div className="max-w-7xl mx-auto mt-24 pt-12 border-t border-white/10 flex justify-between items-center">
          <div className="flex gap-8">
            <a href="https://www.linkedin.com/in/ali-abouelazz-a00197220" className="text-white/70 hover:text-[#9F8E6D] transition-colors duration-300">
              <Linkedin className="w-5 h-5" />
            </a>
          </div>
        </div>
      </section>
    </div>
  );
};

export default DownloadsPage;