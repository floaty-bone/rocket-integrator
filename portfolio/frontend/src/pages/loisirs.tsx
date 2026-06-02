import React, { useState, useEffect } from 'react';
import { Linkedin } from 'lucide-react';
import { Link } from 'react-router-dom';
import RippleMesh from '../components/RippleMesh';
//test
const CentreInteret = () => {
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
            <Link to="/rocketDemo" className="hover:text-[#9F8E6D] transition-colors duration-300">ROCKET DEMO</Link>
            <a href='#' onClick={handleContactClick} className="hover:text-[#9F8E6D] transition-colors duration-300">CONTACT</a>
          </div>
        </div>
      </nav>

      {/* Interests Section */}
      <section className="relative z-10 text-white py-32 px-24 pt-48">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-4xl font-light mb-16 text-center">Interests</h2>

          {/* Surfing Section */}
          <div className="mb-24">
            <div className="grid grid-cols-2 gap-16 items-center">
              <div>
                <img
                  src="/ali-portfolio/images-videos/me_surfing.jpg"
                  alt="Surfing"
                  className="w-full h-[400px] object-cover rounded-lg object-left"
                />
              </div>
              <div>
                <p className="text-xl text-gray-300 leading-relaxed">
                  I got into surfing through my brother. That's me at Safi Point, better known as Le Jardin. A mystical spot that comes alive only a handful of times a year. Caught right before I set up for a barrel and got absolutely worked.
                </p>
              </div>
            </div>
          </div>

          {/* Guitar Section */}
          <div>
            <div className="grid grid-cols-2 gap-16 items-center">
              <div>
                <video
                  className="w-full h-[400px] object-cover rounded-lg"
                  controls
                  playsInline
                >
                  <source src="/ali-portfolio/images-videos/video.mp4" type="video/mp4" />
                  {/* Fallback image in case video doesn't load */}
                  <img
                    src="/api/placeholder/800/600"
                    alt="Playing Guitar"
                    className="w-full h-full object-cover"
                  />
                </video>
              </div>
              <div>
                <p className="text-xl text-gray-300 leading-relaxed">
                  I'm more into jazz fusion now, but back then I was all about classic rock and blues. The video on the left is from 2016, where I'm playing 'The Loner' by Gary Moore.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section className="relative z-10 py-32 px-24">
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

export default CentreInteret;