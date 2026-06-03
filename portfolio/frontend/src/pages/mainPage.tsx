import React, { useState, useEffect } from 'react';
import { ChevronRight, Linkedin } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';
import RippleMesh from '../components/RippleMesh';

const HomePage = () => {
  const [scrollPosition, setScrollPosition] = useState(0);
  const [navVisible, setNavVisible] = useState(true);
  const location = useLocation();

  useEffect(() => {
    if ((location.state as any)?.scrollToContact) {
      setTimeout(() => document.getElementById('contact')?.scrollIntoView({ behavior: 'smooth' }), 100);
    }
  }, [location.state]);

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
      <div className="fixed inset-0 z-0" style={{ background: 'rgba(6,6,10,0.75)' }} />

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
            <Link to="/rocketDemo" className="hover:text-[#9F8E6D] transition-colors duration-300">LQR CONTROL DEMO</Link>
            <a href='#' onClick={handleContactClick} className="hover:text-[#9F8E6D] transition-colors duration-300">CONTACT</a>
          </div>
        </div>
      </nav>
      {/* Hero banner */}
      <div className="relative w-full h-[100vh] z-10">
        <div className="relative h-full flex flex-col justify-center px-24 max-w-4xl">
          <p className="text-base text-gray-200 mb-12 leading-relaxed">
            Hello, I'm Ali Abouelazz
            <br /><br />
            Graduate mechanical engineer from Grenoble INP, Product Engineering specialization (IdP). Multidisciplinary training covering the full product development cycle: from conceptualization to physical prototyping, including CAD modeling, numerical simulation, and control systems.
          </p>
          <Link to="/downloadsPage"
            className="group flex items-center gap-2 text-lg hover:text-[#9F8E6D] transition-all duration-300">
            Technical Portfolio
            <ChevronRight className="w-5 h-5 transform group-hover:translate-x-1 transition-transform duration-300" />
          </Link>
        </div>
      </div>
      {/* Quote Section */}
      <header className="relative z-10 h-screen">
        {/* Hero Content */}
        <div className="relative h-full flex flex-col justify-center px-24 max-w-4xl">
          <blockquote className="relative pl-6 border-l-4 border-gray-300 italic text-base text-gray-200 mb-12 leading-relaxed">
            "I am putting myself to the fullest possible use, which is all I think that any conscious entity can ever hope to do"
            <cite className="block text-sm not-italic text-gray-400 mt-2">— HAL 9000, 2001: A Space Odyssey</cite>
          </blockquote>
        </div>
      </header>

      {/* Contact Section */}
      <section id="contact" className="relative z-10 py-32 px-24">
        <div className="max-w-7xl mx-auto flex justify-between items-start">
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

export default HomePage;