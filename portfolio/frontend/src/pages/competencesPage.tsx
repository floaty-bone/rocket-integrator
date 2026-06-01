import React, { useState, useEffect } from 'react';
import { ChevronLeft, ChevronRight, Linkedin } from 'lucide-react';
import { Link } from 'react-router-dom';
import RippleMesh from '../components/RippleMesh';

const CompetencesPage = () => {
  const [scrollPosition, setScrollPosition] = useState(0);
  const [navVisible, setNavVisible] = useState(true);
  const [currentSkillIndex, setCurrentSkillIndex] = useState(0);

  const skills = [
    {
      name: 'Computer-Aided Design (CAD)',
      image: '/ali-portfolio/images-videos/skillsImages/1.png',
      description: 'Expertise in advanced design techniques (CNC machining, 3D printing, casting)'
    },
    {
      name: 'Finite Element Analysis (FEA)',
      image: '/ali-portfolio/images-videos/skillsImages/4.png',
      description: 'Advanced simulation and structural integrity analysis'
    },
    {
      name: 'Fluid Flow + Thermal Analysis',
      image: '/ali-portfolio/images-videos/skillsImages/2.png',
      description: 'Fluid dynamics and heat transfer'
    },
    {
      name: 'Software Development',
      image: '/ali-portfolio/images-videos/skillsImages/3.png',
      description: 'Software development in C++ and Python'
    },
    {
      name: 'Control Systems Design',
      image: '/ali-portfolio/images-videos/skillsImages/controlSystems.png',
      description: 'Design of automation and control systems'
    }
  ];

  const software = [
    { name: 'Ansys Workbench', logo: '/ali-portfolio/images-videos/softwareImages/ansys.png' },
    { name: 'Matlab Simulink', logo: '/ali-portfolio/images-videos/softwareImages/simulink.jpg' },
    { name: 'C++', logo: '/ali-portfolio/images-videos/softwareImages/CPP.png' }, // Fixed casing from search
    { name: 'Python', logo: '/ali-portfolio/images-videos/softwareImages/python.png' },
    { name: 'Creo', logo: '/ali-portfolio/images-videos/softwareImages/creo.svg.png' },
{ name: 'CATIA', logo: '/ali-portfolio/images-videos/softwareImages/catia.png' },
    { name: 'SolidWorks', logo: '/ali-portfolio/images-videos/softwareImages/solidWorks.png' }
  ];

  const languages = [
    { name: 'French', level: 'Native language' },
    { name: 'Arabic', level: 'Native language' },
    { name: 'English', level: 'C2 Level' }
  ];

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

  const handleSkillNext = () => {
    setCurrentSkillIndex((prev) =>
      (prev + 1) % skills.length
    );
  };

  const handleSkillPrev = () => {
    setCurrentSkillIndex((prev) =>
      prev === 0 ? skills.length - 1 : prev - 1
    );
  };

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

      {/* Skills Content */}
      <div className="relative z-10 min-h-screen px-24 pt-32">
        <div className="flex flex-col justify-center">
          <div className="w-full max-w-6xl mx-auto">
            {/* Skills Section */}
            <section className="mb-16">
              <h2 className="text-4xl font-light mb-12 text-white">Technical Skills</h2>
              <div className="relative flex items-center justify-center">
                <button
                  onClick={handleSkillPrev}
                  className="absolute left-0 z-10 bg-white/10 hover:bg-white/20 p-2 rounded-full transition-colors"
                >
                  <ChevronLeft className="w-8 h-8 text-white" />
                </button>
                <div className="flex items-center justify-center w-full">
                  <div className="w-full max-w-4xl grid grid-cols-2 gap-8 items-center">
                    <div className="bg-white/10 border border-white/20 rounded-lg overflow-hidden">
                      <img
                        src={skills[currentSkillIndex].image}
                        alt={skills[currentSkillIndex].name}
                        className="w-full h-64 object-cover"
                      />
                    </div>
                    <div className="p-8">
                      <h3 className="text-3xl mb-4">{skills[currentSkillIndex].name}</h3>
                      <p className="text-gray-300">{skills[currentSkillIndex].description}</p>
                    </div>
                  </div>
                </div>
                <button
                  onClick={handleSkillNext}
                  className="absolute right-0 z-10 bg-white/10 hover:bg-white/20 p-2 rounded-full transition-colors"
                >
                  <ChevronRight className="w-8 h-8 text-white" />
                </button>
              </div>
            </section>

            {/* Software Section */}
            <section className="mb-16">
              <h2 className="text-4xl font-light mb-12 text-white">Tools & Software</h2>
              <div className="grid grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-4">
                {software.map((item) => (
                  <div
                    key={item.name}
                    className="flex flex-col items-center bg-white/10 p-4 rounded-lg"
                  >
                    <img
                      src={item.logo}
                      alt={item.name}
                      className="w-19 h-16 mb-3 object-contain"
                    />
                    <span className="text-sm">{item.name}</span>
                  </div>
                ))}
              </div>
            </section>

            {/* Languages Section */}
            <section>
              <h2 className="text-4xl font-light mb-12 text-white">Languages</h2>
              <div className="grid grid-cols-3 gap-8">
                {languages.map((lang) => (
                  <div
                    key={lang.name}
                    className="bg-white/10 border border-white/20 rounded-lg p-6 text-center"
                  >
                    <h3 className="text-2xl mb-2">{lang.name}</h3>
                    <p className="text-gray-300">{lang.level}</p>
                  </div>
                ))}
              </div>
            </section>
          </div>
        </div>
      </div>

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

export default CompetencesPage;