import React, { useState, useEffect } from 'react';
import { Linkedin } from 'lucide-react';
import { Link } from 'react-router-dom';
import RippleMesh from '../components/RippleMesh';

const ExperienceEducationPage = () => {
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

      {/* Navigation - identical to main page */}
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

      {/* Hero Section with Experience & Education */}
      <header className="relative z-10 min-h-screen pt-24" style={{ top: "2rem" }}>
        <div className="max-w-7xl mx-auto px-24">

          {/* Education Section */}
          <div className="mb-16">
            <h3 className="text-3xl font-light mb-8 border-b border-white/20 pb-4">Education</h3>

            {/* Grenoble INP Degree */}
            <div className="flex mb-10">
              <div className="mr-4">
                <img
                  src="/ali-portfolio/images-videos/inp.png"
                  alt="Small thumbnail"
                  style={{ width: '120px', height: '40px' }}
                />
              </div>
              <div className="mb-12">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="text-2xl mb-2">Engineering Degree – Grenoble INP</h4>
                    <p className="text-gray-400 mb-4">2022-2025</p>
                  </div>
                  <div className="text-right">
                    <p className="text-[#9F8E6D]">Mechanical Engineering</p>
                  </div>
                </div>
                <p className="text-gray-300 leading-relaxed">
                  Multidisciplinary mechanical engineering training at Grenoble INP – Génie Industriel, Product Engineering specialization (IdP), covering
                  the full product development cycle: from conceptualization to physical prototyping.
                  Proficiency in CAD modeling (Creo, CATIA), numerical simulation
                  (ANSYS Mechanical, Fluent), and MATLAB Simulink for system modeling and control.
                </p>
              </div>
            </div>
            {/* Preparatory Classes */}
            <div className="flex mb-10">
              <div className="mr-4">
                <img
                  src="/ali-portfolio/images-videos/prepa.jpg"
                  alt="Small thumbnail"
                  style={{ width: '50px', height: '45px' }}
                />
              </div>
              <div>
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="text-2xl mb-2">Preparatory Classes</h4>
                    <p className="text-gray-400 mb-4">2020-2022</p>
                  </div>
                  <div className="text-right">
                    <p className="text-[#9F8E6D]">Mathematics & Physics (MPSI/MP)</p>
                  </div>
                </div>
                <p className="text-gray-300 leading-relaxed">
                  Two intensive years in mathematics, physics, and foundational engineering, developing
                  a solid understanding of core scientific principles.
                </p>
              </div>
            </div>
          </div>

          {/* Professional Experience Section */}
          <div className="relative min-h-screen pt-15" style={{ top: "4rem" }}>
            <h3 className="text-3xl font-light mb-8 border-b border-white/20 pb-4">Professional Experience</h3>

            {/* AI Agent Project */}
            <div className="flex mb-10">
              <div className="mr-4">
                <img
                  src="/ali-portfolio/images-videos/yc.svg"
                  alt="Y Combinator logo"
                  style={{ width: '170px', height: 'auto', objectFit: 'contain' }}
                />
              </div>
              <div className="mb-12">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="text-2xl mb-2">Implementation and Deployment of a Resident Autonomous AI Agent</h4>
                    <p className="text-gray-400 mb-4">09/2025 – 02/2026 | Y Combinator February Batch Applicant</p>
                  </div>
                  <div className="text-right">
                    <p className="text-[#9F8E6D]">AI & Software Development</p>
                  </div>
                </div>
                <p className="text-gray-300 leading-relaxed">
                  Designed a resident AI agent running persistently on the user's machine (Linux, Windows, macOS)
                  inside a persistent Firecracker MicroVM. Full system access via MCP tools (bash, files, CDP browser),
                  continuous cross-session memory via ledger with FIFO compression. 3-tier architecture
                  (Orchestrator → Worker → Cron Agent) with sub-agent delegation and autonomous scheduling.
                  Unified LLM client (Claude, OpenAI, Gemini, Grok). Stack: Python, asyncio, Firecracker, cdp-use, MCP.{' '}
                  <a href="https://github.com/floaty-bone/maxent" className="text-[#9F8E6D] hover:underline">GitHub</a>
                </p>
              </div>
            </div>

            {/* Caterpillar Stage PFE */}
            <div className="flex mb-10">
              <div className="mr-4">
                <img
                  src="/ali-portfolio/images-videos/Cat_Logo.png"
                  alt="Caterpillar logo"
                  style={{ width: '90px', height: 'auto', objectFit: 'contain' }}
                />
              </div>
              <div className="mb-12">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="text-2xl mb-2">Final Year Internship – Optimization of Caterpillar D5 Bulldozer Track Tensioner</h4>
                    <p className="text-gray-400 mb-4">02/2025 – 09/2025 | Caterpillar</p>
                  </div>
                  <div className="text-right">
                    <p className="text-[#9F8E6D]">Reliability Analysis & Design</p>
                  </div>
                </div>
                <p className="text-gray-300 leading-relaxed">
                  Reliability analysis of the Caterpillar D5 track tensioner using field data (warranty claims),
                  identification and prioritization of critical failures via DMAIC/FMEA methodology.
                  Design validation through FEA simulation in ANSYS Mechanical and 3D CAD modeling
                  of improvement concepts (Creo Parametric). Tools: Python, ANSYS, Creo.
                </p>
              </div>
            </div>

            {/* General Electric Internship */}
            <div className="flex mb-10">
              <div className="mr-4">
                <img
                  src="/ali-portfolio/images-videos/ge.png"
                  alt="General Electric logo"
                  style={{ width: '170px', height: 'auto', objectFit: 'contain' }}
                />
              </div>
              <div className="mb-12">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="text-2xl mb-2">Assistant Engineer Internship</h4>
                    <p className="text-gray-400 mb-4">05/2024 – 08/2024 | General Electric, Lyon</p>
                  </div>
                  <div className="text-right">
                    <p className="text-[#9F8E6D]">Design & Development</p>
                  </div>
                </div>
                <p className="text-gray-300 leading-relaxed">
                  Designed a compliance testing station for high-voltage live tank circuit breakers,
                  targeting contact crown finger conformity. Automatic acquisition of stroke and force data,
                  drafting of technical specifications, verification via static calculations
                  and ANSYS simulations. C++ development of the graphical interface (ImGui) and
                  data logging to a local server (MySQL). Secondary task: verification calculations and
                  sizing of connecting rods and SF6 chambers for GCB circuit breakers.
                </p>
              </div>
            </div>

            {/* Alstom School Project */}
            <div className="flex mb-10">
              <div className="mr-4">
                <img
                  src="/ali-portfolio/images-videos/alstom.png"
                  alt="Alstom logo"
                  style={{ width: '130px', height: 'auto', objectFit: 'contain' }}
                />
              </div>
              <div className="mb-12">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="text-2xl mb-2">School Project</h4>
                    <p className="text-gray-400 mb-4">04/2024 – 05/2024 | Alstom, Lyon</p>
                  </div>
                  <div className="text-right">
                    <p className="text-[#9F8E6D]">Simulation & Design</p>
                  </div>
                </div>
                <p className="text-gray-300 leading-relaxed">
                  Design and optimization of a mechanical test bench reproducing periodic radial
                  and axial loads on a bearing integrated into a new generator line.
                  Instrumentation, experimental data acquisition, and fatigue simulation (ANSYS)
                  to estimate service life and qualify the product.
                </p>
              </div>
            </div>

            {/* Sabca Internship */}
            <div className="flex mb-10">
              <div className="mr-4">
                <img
                  src="/ali-portfolio/images-videos/sabca.png"
                  alt="Sabca logo"
                  style={{ width: '100px', height: '30px' }}
                />
              </div>
              <div className="mb-12">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="text-2xl mb-2">Operator Internship</h4>
                    <p className="text-gray-400 mb-4">07/2023 – 08/2023 | Sabca (Pilatus PC-12)</p>
                  </div>
                  <div className="text-right">
                    <p className="text-[#9F8E6D]">Aerospace Assembly</p>
                  </div>
                </div>
                <p className="text-gray-300 leading-relaxed">
                  Followed a tooling layout plan and monitored the placement of jigs,
                  lifting structures, tables, and equipment to ensure industrial layout compliance
                  on the Pilatus PC-12 aircraft assembly line.
                </p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Contact Section */}
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

export default ExperienceEducationPage;