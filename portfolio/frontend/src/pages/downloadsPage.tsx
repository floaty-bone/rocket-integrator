import React, { useState, useEffect } from 'react';
import { Download, Linkedin, ChevronRight, ChevronLeft, ExternalLink } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import RippleMesh from '../components/RippleMesh';

const BASE = '/ali-portfolio/portfolio-images';

// ─── KaTeX helpers ────────────────────────────────────────────────────────────

function Eq({ children, display = false }: { children: string; display?: boolean }) {
  const html = katex.renderToString(children, { throwOnError: false, displayMode: display });
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

function EqBlock({ children }: { children: string }) {
  const html = katex.renderToString(children, { throwOnError: false, displayMode: true });
  return <div className="my-3 overflow-x-auto" dangerouslySetInnerHTML={{ __html: html }} />;
}

// ─── Layout primitives ────────────────────────────────────────────────────────

function Subsection({ title }: { title: string }) {
  return <h4 className="text-base font-semibold text-[#E8C97A] mt-5 mb-2">{title}</h4>;
}

function Subsubsection({ title }: { title: string }) {
  return <h5 className="text-sm font-semibold text-white/80 mt-4 mb-1.5 uppercase tracking-wide">{title}</h5>;
}

function Para({ children }: { children: React.ReactNode }) {
  return <p className="text-sm text-gray-300 leading-relaxed mb-3">{children}</p>;
}

function Outcome({ accent, title, children }: { accent: string; title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg p-4 mt-5" style={{ background: `${accent}12`, border: `1px solid ${accent}40` }}>
      <p className="text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: accent }}>{title}</p>
      <div className="text-sm text-gray-300 leading-relaxed space-y-1">{children}</div>
    </div>
  );
}

function Note({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-lg p-4 mt-4 bg-white/5 border border-white/15">
      <p className="text-xs font-semibold uppercase tracking-widest text-gray-400 mb-2">Note</p>
      <div className="text-sm text-gray-300 leading-relaxed">{children}</div>
    </div>
  );
}

function DataTable({ caption, rows, head }: { caption: string; rows: string[][]; head: string[] }) {
  return (
    <div className="my-4">
      <div className="overflow-x-auto rounded-md border border-white/15">
        <table className="w-full text-xs text-gray-300">
          <thead>
            <tr className="border-b border-white/15 bg-white/5">
              {head.map((h) => <th key={h} className="px-3 py-2 text-left font-semibold text-white/80">{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className="border-b border-white/10 last:border-0">
                {row.map((cell, j) => <td key={j} className="px-3 py-2">{cell}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-gray-500 mt-1.5 italic">{caption}</p>
    </div>
  );
}

function Fig({ src, caption, small }: { src: string; caption: string; small?: boolean }) {
  return (
    <div className={`my-4 ${small ? 'flex flex-col items-center' : ''}`}>
      <div className={`rounded-md overflow-hidden bg-black/30 border border-white/10 ${small ? 'inline-block' : 'w-full'}`}>
        <img src={src} alt={caption} className={`h-auto block ${small ? 'max-h-52 max-w-sm w-auto' : 'w-full'}`} />
      </div>
      <p className={`text-xs text-gray-500 mt-1.5 italic ${small ? 'text-center max-w-sm' : ''}`}>{caption}</p>
    </div>
  );
}

function FigRow({ items }: { items: { src: string; caption?: string }[] }) {
  return (
    <div className="my-4">
      <div className={`grid gap-3`} style={{ gridTemplateColumns: `repeat(${items.length}, 1fr)` }}>
        {items.map((item, i) => (
          <div key={i} className="rounded-md overflow-hidden bg-black/30 border border-white/10">
            <img src={item.src} alt={item.caption ?? ''} className="w-full h-auto object-contain max-h-56" />
          </div>
        ))}
      </div>
      {items.some(i => i.caption) && (
        <p className="text-xs text-gray-500 mt-1.5 italic">{items.map(i => i.caption).filter(Boolean).join(' — ')}</p>
      )}
    </div>
  );
}

// ─── Image gallery (used where LaTeX had multiple figures in a section) ────────

function Gallery({ images, captions }: { images: string[]; captions?: string[] }) {
  const [idx, setIdx] = useState(0);
  if (images.length === 0) return null;
  const caption = captions?.[idx];
  return (
    <div className="my-4">
      <div className="relative rounded-md overflow-hidden bg-black/30 border border-white/10" style={{ minHeight: '180px' }}>
        <img src={images[idx]} alt={caption ?? ''} className="w-full h-auto object-contain max-h-80" />
        {images.length > 1 && (
          <>
            <button onClick={() => setIdx((i) => (i - 1 + images.length) % images.length)}
              className="absolute left-2 top-1/2 -translate-y-1/2 p-1 rounded-full bg-black/60 hover:bg-black/80">
              <ChevronLeft className="w-4 h-4 text-white" />
            </button>
            <button onClick={() => setIdx((i) => (i + 1) % images.length)}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-1 rounded-full bg-black/60 hover:bg-black/80">
              <ChevronRight className="w-4 h-4 text-white" />
            </button>
            <div className="absolute bottom-2 left-1/2 -translate-x-1/2 flex gap-1">
              {images.map((_, i) => (
                <button key={i} onClick={() => setIdx(i)}
                  className="w-1.5 h-1.5 rounded-full transition-colors"
                  style={{ background: i === idx ? '#9F8E6D' : 'rgba(255,255,255,0.35)' }} />
              ))}
            </div>
          </>
        )}
      </div>
      {caption && <p className="text-xs text-gray-500 mt-1.5 italic">{caption}</p>}
    </div>
  );
}

// ─── Major section (top-level accordion) ─────────────────────────────────────

function MajorSection({ title, children }: { title: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border rounded-xl overflow-hidden transition-colors duration-300"
      style={{ borderColor: open ? 'rgba(255,255,255,0.22)' : 'rgba(255,255,255,0.09)', background: open ? 'rgba(255,255,255,0.02)' : 'rgba(255,255,255,0.01)' }}>
      <button className="w-full flex items-center justify-between px-8 py-6 text-left" onClick={() => setOpen(o => !o)}>
        <h2 className="text-2xl font-light tracking-wide text-white">{title}</h2>
        <ChevronRight className="w-5 h-5 flex-shrink-0 transition-transform duration-300 text-white/50"
          style={{ transform: open ? 'rotate(90deg)' : 'rotate(0deg)' }} />
      </button>
      {open && <div className="px-6 pb-6 pt-1 space-y-3">{children}</div>}
    </div>
  );
}

// ─── Sub-section (nested accordion inside a MajorSection) ────────────────────

function SubSection({
  label, title, accent, children,
}: {
  label: string; title: string; accent: string; children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border rounded-lg overflow-hidden transition-colors duration-300"
      style={{ borderColor: open ? `${accent}45` : 'rgba(255,255,255,0.08)', background: open ? 'rgba(255,255,255,0.03)' : 'transparent' }}>
      <button className="w-full flex items-center justify-between px-6 py-5 text-left" onClick={() => setOpen(o => !o)}>
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest mb-0.5" style={{ color: accent }}>{label}</p>
          <h3 className="text-lg font-light text-white">{title}</h3>
        </div>
        <ChevronRight className="w-4 h-4 flex-shrink-0 transition-transform duration-300"
          style={{ transform: open ? 'rotate(90deg)' : 'rotate(0deg)', color: accent }} />
      </button>
      {open && <div className="px-6 pb-7 pt-1">{children}</div>}
    </div>
  );
}

// ─── Rocket demo card ─────────────────────────────────────────────────────────

function RocketDemoCard() {
  const navigate = useNavigate();
  return (
    <div
      className="border rounded-xl overflow-hidden transition-colors duration-300 cursor-pointer group"
      style={{ borderColor: 'rgba(159,142,109,0.30)', background: 'rgba(159,142,109,0.04)' }}
      onClick={() => navigate('/rocketDemo')}
    >
      <div className="p-7 flex flex-col md:flex-row items-start md:items-center gap-6">
        <div className="flex-1 min-w-0">
          <p className="text-xs font-semibold uppercase tracking-widest mb-1 text-[#9F8E6D]">Live 3D Demo — Personal Study</p>
          <h3 className="text-xl font-light text-white mb-2">Starship Booster Landing</h3>
          <p className="text-sm text-gray-300 leading-relaxed max-w-xl">
            Experimenting with LQR full-state feedback control — live 3D simulation launch demo.
          </p>
        </div>
        <div className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-light flex-shrink-0 transition-all duration-300"
          style={{ border: '1px solid rgba(159,142,109,0.50)', color: '#9F8E6D' }}
          onMouseEnter={e => { e.currentTarget.style.background = '#9F8E6D'; e.currentTarget.style.color = 'white'; }}
          onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = '#9F8E6D'; }}>
          <span>Launch Demo</span>
          <ExternalLink className="w-3.5 h-3.5" />
        </div>
      </div>
    </div>
  );
}

// ─── GE Vernova content ───────────────────────────────────────────────────────

function GEContent() {
  return (
    <>
      <Subsection title="Project 1: Stiffness Measurement Station" />

      <Subsubsection title="Objective" />
      <Para>
        Design (mechanical and software components) of a stiffness measurement station for the finger contact springs of
        the new series of live tank circuit breakers. The purpose of this station is to ensure the conformity of the
        finger contact springs after assembly in the housing.
      </Para>

      <Subsubsection title="Execution" />
      <Para>
        My responsibility was to develop the station in its entirety, both mechanically and in terms of software. I
        began by defining requirements with the various stakeholders (quality department, operators, design office).
        These needs were translated into system-level requirements, then broken down into subsystem-level requirements
        and further into component-level requirements (top-down design approach).
      </Para>
      <Para>
        Once the component-level requirements were defined, I began the design phase while simultaneously working with
        my supervisor to identify the simulations needed to validate the measurement accuracy requirements. After
        dimensional validation, I produced the engineering drawings, which were then sent to the procurement department.
      </Para>
      <Para>
        The software component was developed in parallel. I created a graphical user interface for data visualisation
        and control of the force sensor, then synchronised it with the site database for data logging.
      </Para>

      <FigRow items={[
        { src: `${BASE}/image9.png`, caption: 'Mechanical design of the measurement station' },
        { src: `${BASE}/image10.png`, caption: 'Software interface and data acquisition system' },
      ]} />

      <Subsection title="Project 2: Power Transmission Shaft Sizing" />

      <Subsubsection title="Objective" />
      <Para>
        Sizing of a redesigned power transmission shaft for the opening mechanism of GCB (Ground Circuit Breaker)
        circuit breakers. The previous system was costly and sensitive to coaxiality defects.
      </Para>

      <Subsubsection title="Execution" />
      <Para>
        I sized a new shaft design by first simulating the various proposals made by a colleague. I iterated over
        different dimensions and made minor modifications to achieve the required safety factor without exceeding it.
        The redesign aimed to simplify machining operations and, most importantly, to reduce the amount of material
        used, which represented a significant cost for the company.
      </Para>

      <Fig src={`${BASE}/image99.png`} caption="FEA analysis of the initial design" />

      <Subsection title="Project 3: Stress Analysis of SF6 Gas Chamber" />

      <Subsubsection title="Objective" />
      <Para>
        Determine the mechanical stress in the SF6 gas chamber due to the gas pressure, which was increased in order to
        extend the service life of the tulip contacts. The objective is to ensure that the safety factor remains above 5
        as specified in the design requirements (CDC 912625/CODAP).
      </Para>

      <Subsubsection title="Execution" />
      <Para>
        The load was applied to the inner walls to simulate a static pressure of <Eq>{'p = 1.6\\ \\text{MPa}'}</Eq>.
        Material used: A-S7G03.
      </Para>

      <Fig src={`${BASE}/image1.png`} caption="FEA model of the gas chamber" />
    </>
  );
}

// ─── Caterpillar content ──────────────────────────────────────────────────────

function CatContent() {
  return (
    <>
      <Subsection title="Objective and Context" />
      <Para>
        The Caterpillar D5 bulldozer had been accumulating field failures on its chain tensioner assembly: grease leaks,
        cracked plugs, worn retainer plates, premature piston-seal failures. The product-support team had flagged the
        problem but the failure modes had never been quantified, and the dominant root causes were still unclear. My
        internship mission was to close that loop: diagnose the system end-to-end, validate the diagnosis with
        simulation, and deliver redesign concepts that addressed the root causes while remaining compatible with
        manufacturing and assembly constraints.
      </Para>
      <Para>
        The project followed a full <strong className="text-white">DMAIC cycle</strong>. Scope was restricted to the D5
        variants (D5 LGP, D5 XL) and specifically to the tensioner group. The D4, D6, D7, John Deere 700L and Komatsu
        D51 PX systems were used as benchmark references. Deliverables were 3D CAD redesigns validated in FEA, not
        physical prototypes. Tools: Creo Parametric for CAD, ANSYS Mechanical for structural FEA, Python for data
        processing.
      </Para>

      <Fig
        src={`${BASE}/cat/13.png`}
        caption="D5 track tensioner: cross-section showing the main components involved in the failure modes (spring tube, piston, piston seal, retainer, cylinder, plug). Grease pressure acts on the piston base to tension the track; the plug seals the rear of the cylinder and is bolted to the housing."
      />
      <FigRow items={[{ src: `${BASE}/cat/14.png` }, { src: `${BASE}/cat/15.png` }]} />

      <Subsection title="Automated Failure-Mode Classification of Warranty Reports" />
      <Para>
        Caterpillar maintains a centralised warranty database: every time a dealer repairs a machine under warranty,
        they log the failed part, the machine hours, a short <em>comment</em> field and a longer <em>claim story</em>.
        Filtering by the tensioner part numbers returned records{' '}
        <strong className="text-white">in the order of thousands</strong> — a dataset large enough to make statistical
        analysis meaningful, but far too large to read manually. The real obstacle was that dealers worldwide write in
        their native language: English, German, Polish, Japanese, Spanish, sometimes with abbreviations and regional
        shorthand. Per-component <em>frequency</em> was easy to compute from the part number alone, but extracting the
        actual <em>failure mode</em> from free text required something smarter.
      </Para>
      <Para>
        My first attempt used the Gemini API: feed each concatenated{' '}
        <code className="text-xs bg-white/10 px-1 rounded">comment + claim_story</code> to the model with a system
        prompt asking it to pick one mode from a fixed list. It worked well, but every call sent customer-written
        warranty text to a third-party server. Even though the raw text did not mention "Caterpillar" or part numbers,
        sending it out over an API was not acceptable for this use case. Deploying a local LLM server (Ollama-style)
        would have solved the privacy issue but required IT approvals I was not going to get in time.
      </Para>
      <Para>
        I pivoted to a lighter, fully local approach based on{' '}
        <strong className="text-white">semantic similarity with Sentence-BERT</strong>. SBERT embeds a sentence into a
        dense high-dimensional vector whose geometry captures semantic meaning — two sentences expressing the same idea
        in different words (or different languages) land close together. I embedded each of the eight candidate failure
        modes (<em>grease leak, broken threads, detached part, loose part, broken/cracked part, corroded part, clogged,
        damaged threads/grooves</em>) once, then embedded every claim story, and classified each one by the argmax of
        cosine similarity:
      </Para>
      <EqBlock>{`\\mathrm{FM}(t) \\;=\\; \\arg\\max_i \\; \\frac{\\mathbf{v}_t \\cdot \\mathbf{v}_{\\mathrm{MD}_i}}{\\|\\mathbf{v}_t\\|\\,\\|\\mathbf{v}_{\\mathrm{MD}_i}\\|}`}</EqBlock>
      <Para>
        A similarity threshold was applied below which the case was marked null, so borderline or uninformative claim
        stories would not pollute the statistics. The{' '}
        <code className="text-xs bg-white/10 px-1 rounded">all-MiniLM-L6-v2</code> model runs comfortably on CPU and
        handled the multilingual content well enough that manual translation was unnecessary. The entire pipeline ran on
        my workstation — no data left the machine. I validated the output against 50 manually-labelled samples and the
        classifier hit <strong className="text-white">48/50 = 96% accuracy</strong>, which was well within acceptable
        bounds for prioritisation work.
      </Para>
      <Para>
        Crossing the resulting failure-mode distribution with each component's warranty cost revealed a clear ranking of
        critical failures: leakage at the piston/seal interface, leakage at the fill-valve/cylinder interface, plug
        loosening and cracking, and leakage at the relief-valve interface. Equally striking was what the data{' '}
        <em>did not</em> contain: zero reports of the intentional fuse system actuating, despite multiple plug-cracking
        cases. That contradiction became the focus of the next phase.
      </Para>

      <FigRow items={[{ src: `${BASE}/cat/19.png` }, { src: `${BASE}/cat/20.png` }]} />
      <Fig src={`${BASE}/cat/21.png`}
        caption="Failure-mode distribution per component after automated classification of the warranty records. Grease leak dominates across seals, valves and plug." />

      <Subsection title="FEA of the Fuse System: Why Plugs Were Cracking" />
      <Para>
        The tensioner is built with an intentional mechanical fuse — a deformable steel bar backed by an O-ring —
        designed to open a leak path <em>before</em> any other component reaches its yield limit. If the fuse was doing
        its job, cracked plugs should not exist in the warranty record. They did. Something in the actual stress
        response of the assembly was violating the intent of the design, and I needed an FEA model to find out what.
      </Para>
      <Para>
        I built a coupled model of the fuse assembly in ANSYS Mechanical, applying grease pressure directly to the
        internal surfaces and pre-tensioning the bolts and plug per the design drawings. The first version modelled the
        O-ring explicitly as a hyperelastic body — the theoretically correct choice. In practice the O-ring deformation
        became so large that mesh elements distorted past ANSYS's convergence tolerance, and the simulation crashed
        above 24 pressure units, far below the regime of interest (130–250 units).
      </Para>
      <Para>
        Rather than fight the solver with finer meshes and remeshing hacks, I reformulated the leak criterion{' '}
        <em>geometrically</em>. Per the Parker O-ring Handbook, an elastomeric seal is guaranteed to hold as long as
        its compression exceeds 5.7%, and is guaranteed to leak below 0% (loss of contact). If I remove the O-ring
        from the simulation entirely and track the local vertical displacement{' '}
        <Eq>{'d_{z,k}'}</Eq> of each node along the seal contour, I can compute a per-node compression:
      </Para>
      <EqBlock>{`C_{\\%,k} \\;=\\; \\frac{t_o - t_{c,k}}{t_o}, \\qquad t_{c,k} = d_{z,k} + g_d`}</EqBlock>
      <Para>
        where <Eq>{'t_o'}</Eq> is the free O-ring thickness and <Eq>{'g_d'}</Eq> is the gland depth. Aggregating over
        all contour nodes gives <strong className="text-white">guaranteed sealing</strong> when{' '}
        <Eq>{'\\min_k C_{\\%,k} > 5.7\\%'}</Eq> and{' '}
        <strong className="text-white">certain leakage</strong> when{' '}
        <Eq>{'\\overline{C_\\%} < 0\\%'}</Eq>. With the O-ring removed, the model meshed cleanly and ran stable all
        the way to plastic yield, while still producing a physically meaningful leak prediction. The model also exploited
        the cylindrical symmetry of the assembly, halving the node count and the solve time.
      </Para>
      <Para>
        The results were unambiguous. At roughly 130 pressure units — right around the maximum service pressure recorded
        during bench testing — the plug threads reached 329 MPa, already above the 310 MPa yield limit of the material.
        At that same pressure the fuse plate had barely deformed, and the O-ring compression was still sitting around
        20% everywhere along the contour. Even pushing the simulation all the way to 240 units, the fuse still would not
        open a leak path. <strong className="text-white">The fuse was massively oversized and effectively inactive</strong>,
        leaving the plug as the <em>de facto</em> weakest link — exactly consistent with the warranty record.
      </Para>

      <FigRow items={[{ src: `${BASE}/cat/30.png` }, { src: `${BASE}/cat/31.png` }]} />
      <p className="text-xs text-gray-500 -mt-2 italic mb-4">
        Von Mises stress in the fuse assembly at 130 pressure units. The plug threads reach 329 MPa (yield = 310 MPa)
        while the fuse plate remains elastic — the fuse never triggers before plug failure.
      </p>

      <Subsection title="Fuse Re-sizing" />
      <Para>
        With the fuse proven ineffective, the cheapest corrective action was to keep the same topology but thin down the
        plate until it actually deformed in the right pressure window. The desired window had clear bounds: seal reliably
        above the 130-unit maximum service pressure, and leak reliably below the 220-unit cylinder yield pressure — a
        90-unit band to work inside. (The plug itself is weaker than the cylinder, but its redesign was out of scope for
        this sizing study; once the plug is redesigned to match the cylinder, the same curves still apply.)
      </Para>
      <Para>
        I re-ran the simulation parametrically across four supplier-standard plate thicknesses — 3, 3.5, 4 and 6 mm —
        and for each one extracted the minimum and average O-ring compression along the contact contour as pressure
        ramped. As expected, thinner plates leak earlier and produce wider grey-zone bands. The 6 mm plate essentially
        replicated the current oversized behaviour; the 3 mm plate leaked too early, barely clearing the 130-unit
        service ceiling.
      </Para>

      <Fig src={`${BASE}/cat/35.png`}
        caption="O-ring compression vs. pressure for each plate thickness. Grey bands mark the transition between guaranteed sealing (min C% > 5.7%) and certain leakage (avg C% < 0%)." />

      <Para>
        The <strong className="text-white">3.5 mm plate</strong> was the clean answer: guaranteed sealing up to around
        160 units (comfortably above 130) and guaranteed leakage by around 180 units (comfortably below 220). Part cost
        stayed negligible since the geometry is still a simple stamped plate. One important limitation worth flagging:
        this fuse is designed for static or quasi-static overpressure. A sudden pressure spike — for example a hard
        idler impact — would require a much higher mass-flow evacuation path than the small opening produced by plate
        deformation. For that failure mode a proper relief valve is the right tool, and that informed the concept-level
        redesign.
      </Para>

      <Subsection title="Redesign Concepts and Pugh-Matrix Selection" />
      <Para>
        Three full CAD redesigns were developed in Creo Parametric, each informed by both the warranty analysis and a
        benchmark of the John Deere 700L, Komatsu D51 PX, and CAT D4/D7/TTL tensioners. The recurring themes across
        successful competitor designs were consistent: welded (not threaded) cylinder assembly to eliminate the
        fill-valve leak path, chromed or sleeved piston-cylinder interface to prevent wear-induced seal failure,
        oil-based lubrication of the sliding contact, a positive mechanical alignment between cylinder and frame, and a
        proper relief valve as the overpressure safety device rather than a deformable bar.
      </Para>
      <Para>
        Concept 1 focused on a perfect-guidance architecture: spherical joint at the piston end, a
        spacer-and-sleeve retainer providing two line contacts against a precision-machined cylinder, and an
        interconnected oil chamber lubricating both interfaces. Concept 2 was inspired directly by the John Deere 700L:
        a two-piece welded cylinder, a re-sized fuse plate from the sizing study, and a guide hole catching the piston
        if the chain goes slack. Concept 3 took a different route — it integrates the cylinder body directly into the
        mobile portion of the TRF, uses a two-level alignment (ball joint at one end, pin-in-hole at the other) that
        makes the sub-assembly essentially self-aligning under its own weight during installation, and keeps both valves
        accessible without modifying the TRF hatch door.
      </Para>
      <Para>
        The three concepts were scored against eight criteria in a Pugh matrix using the current D5 as the reference:
        cost, intrusiveness on the existing design, serviceability, operator safety, ease of assembly, behaviour under
        chain-loose scenarios, fuse robustness, and manufacturability. The matrix was intentionally kept qualitative —
        weighting the criteria with specific numerical coefficients would have introduced arbitrary bias given that
        several criteria (assembly ease especially) were the blocking constraints.
      </Para>
      <Para>
        <strong className="text-white">Concept 3 was selected.</strong> It scored positively on cost, serviceability,
        manufacturability, intrusiveness and chain-loose guidance; neutral on assembly ease, where its self-aligning
        geometry directly addresses what had been the single biggest pain point. Concept 2 was blocked by severe
        assembly difficulty inside the cramped TRF, and Concept 1 offered no clear advantage over the reference.
        Concept 3 is now positioned for physical prototyping and bench validation — the remaining step in the DMAIC
        Control phase.
      </Para>

      {/* Concept 1 and 3 side by side, then Concept 2 row */}
      <div className="grid grid-cols-2 gap-4 my-4">
        <div>
          <div className="space-y-2">
            {[`${BASE}/cat/37.png`, `${BASE}/cat/38.png`, `${BASE}/cat/39.png`].map(src => (
              <div key={src} className="rounded overflow-hidden bg-black/30 border border-white/10">
                <img src={src} alt="" className="w-full h-auto object-contain max-h-32" />
              </div>
            ))}
          </div>
          <p className="text-xs text-gray-500 mt-1 italic text-center">Concept 1</p>
        </div>
        <div>
          <div className="space-y-2">
            {[`${BASE}/cat/43.png`, `${BASE}/cat/44.png`, `${BASE}/cat/45.png`].map(src => (
              <div key={src} className="rounded overflow-hidden bg-black/30 border border-white/10">
                <img src={src} alt="" className="w-full h-auto object-contain max-h-32" />
              </div>
            ))}
          </div>
          <p className="text-xs text-gray-500 mt-1 italic text-center">Concept 3</p>
        </div>
      </div>
      <div className="border-t border-white/10 pt-4 mb-4">
        <div className="grid grid-cols-3 gap-3">
          {[`${BASE}/cat/40.png`, `${BASE}/cat/41.png`, `${BASE}/cat/42.png`].map(src => (
            <div key={src} className="rounded overflow-hidden bg-black/30 border border-white/10">
              <img src={src} alt="" className="w-full h-auto object-contain max-h-36" />
            </div>
          ))}
        </div>
        <p className="text-xs text-gray-500 mt-1 italic text-center">Concept 2</p>
      </div>

      <Outcome accent="#E8A020" title="Key Outcomes">
        <ul className="space-y-1">
          <li>· 684 multilingual warranty reports classified automatically at <strong className="text-white">96% accuracy</strong> with a local SBERT pipeline (no data leaves the machine).</li>
          <li>· FEA with a geometric leak criterion proved the existing fuse never activates, and identified the plug as the real failure point — matching field data exactly.</li>
          <li>· Re-sized fuse plate (3.5 mm) delivers a clean pressure window: seals up to ~160, leaks by ~180, well inside the 130–220 safety band.</li>
          <li>· Concept 3 selected via Pugh matrix, addressing all four critical failure modes identified upstream and ready for prototyping.</li>
        </ul>
      </Outcome>
    </>
  );
}

// ─── Personal Studies content ─────────────────────────────────────────────────

function CombustionContent() {
  return (
    <>
      {/* Study 1 */}
      <Subsection title="Study 1: Optimisation of Regenerative Cooling in a Combustion Chamber" />

      <Subsubsection title="Context" />
      <Para>
        During operation, rocket engines — particularly the combustion chamber — are subjected to extremely high
        temperatures. As temperature increases, the mechanical properties of metals degrade, most notably their yield
        strength. To address this issue, two methods exist for controlling the temperature of the chamber:
      </Para>
      <ul className="text-sm text-gray-300 space-y-1.5 mb-3 pl-4">
        <li><strong className="text-white">Ablative cooling:</strong> involves the gradual erosion of the inner chamber
          walls, carrying away heat in the process. However, this method is destructive and assumes the engine will not
          be reused.</li>
        <li><strong className="text-white">Regenerative cooling:</strong> the most commonly used method. It works by
          circulating a cryogenic fluid through channels machined into the combustion chamber wall, enabling efficient
          heat dissipation.</li>
      </ul>

      <Subsubsection title="Study Objective" />
      <Para>
        Evaluate the effect of varying the width <Eq>{'L'}</Eq> and height <Eq>{'H'}</Eq> of straight cooling channels
        on the maximum temperature of the inner combustion chamber wall. The objective is to find an optimum with the
        maximum inner wall temperature as the objective function and <Eq>{'L'}</Eq> and <Eq>{'H'}</Eq> as optimisation
        variables, with:
      </Para>
      <ul className="text-sm text-gray-300 space-y-1 mb-3 pl-4">
        <li><Eq>{'L'}</Eq> between 0.8 and 3.0 mm</li>
        <li><Eq>{'H'}</Eq> between 1.5 and 5.0 mm</li>
        <li>Wall thickness kept constant at 10 mm</li>
        <li>Distance between the channel and the inner wall kept constant at 3 mm</li>
      </ul>

      <Fig small src={`${BASE}/image43.jpg`}
        caption="Cooling channel design (CAD assembly of combustion chamber + nozzle)" />

      <Subsubsection title="Study Parameters" />
      <DataTable
        caption="Simulation parameters"
        head={['Parameter', 'Value']}
        rows={[
          ['Fuel/oxidizer', 'Methane/liquid oxygen'],
          ['Inner wall material', 'CuAgZr'],
          ['O₂/fuel mixture ratio', '3.6'],
          ['Chamber pressure (MPa)', '3'],
          ['Coolant', 'N₂ (liquid nitrogen)'],
          ['Coolant mass flow rate (kg/s)', '1.5'],
          ['Coolant inlet temperature (K)', '70'],
          ['Contraction/expansion ratio', '9/5.5'],
          ['Wall thickness (mm)', '10'],
          ['Channel-to-wall margin thickness (mm)', '3'],
          ['Throat diameter (m)', '0.0274'],
        ]}
      />

      <Subsubsection title="Execution" />
      <Para>
        Initially, the simulation was carried out with an inner wall thickness, cooling channel width, and height of
        1 mm, 1.8 mm, and 3.5 mm, respectively.
      </Para>
      <Para>
        The mesh was generated with inflation layers at the boundary layer of the combustion chamber to capture
        temperature and velocity gradients, and similarly for the cooling channels. A mesh of the solid region was also
        generated.
      </Para>

      <Fig small src={`${BASE}/image44.jpg`} caption="Fluid + solid zone mesh" />

      <Para>After simulation in ANSYS Fluent + ANSYS Thermal, the following results were obtained:</Para>

      <DataTable
        caption="Thermal data of the gas in the combustion chamber at selected cross-sections"
        head={['Axial position (mm)', 'Temp. (K)', 'γ', 'Mach', 'Viscosity (10⁻⁴ Pa·s)', 'Cp (J/mol·K)', 'Pr']}
        rows={[
          ['0', '3448.78', '1.1251', '0', '1.1437', '2278.4', '0.5649'],
          ['20.5', '3442.70', '1.1250', '0.192', '1.1422', '2278.0', '0.5651'],
          ['47.3', '3287.31', '1.1213', '1.000', '1.1040', '2267.5', '0.5716'],
          ['79.8', '2783.91', '1.1125', '2.332', '0.9785', '2225.3', '0.5940'],
          ['111.64', '2648.67', '1.1122', '2.635', '0.9444', '2210.7', '0.6000'],
        ]}
      />

      <Fig small src={`${BASE}/image49.png`}
        caption="Temperature distribution in the combustion chamber wall (cross-section taken between two channels)" />

      <Para>
        The temperature peak is reached just before the narrowest cross-section, near the contraction zone of the
        chamber throat.
      </Para>
      <Para>
        After varying parameters <Eq>{'L'}</Eq> and <Eq>{'H'}</Eq> in increments of 0.01 mm (between 1.5 and 5.0 mm
        for height and 0.8 and 3.0 mm for width), the following results were obtained:
      </Para>

      <Fig small src={`${BASE}/image50.png`}
        caption="Maximum combustion chamber temperature as a function of L and H" />

      <Subsubsection title="Conclusion" />
      <Outcome accent="#9F8E6D" title="Optimal Result">
        <Para>
          The optimum is reached at <Eq>{'H = 5\\ \\text{mm}'}</Eq> and <Eq>{'L = 0.8\\ \\text{mm}'}</Eq>, with a
          minimum temperature of <strong className="text-white">586.6 K</strong>.
        </Para>
      </Outcome>
      <Para>
        <strong className="text-white">Interpretation:</strong> When the channel width is increased, the coolant absorbs
        a large amount of heat at the inlet of the combustion chamber due to the large exchange surface area, in a
        region where its effect is not fully needed. By the time it reaches the chamber throat, where temperatures are
        highest, the coolant has already reached a significant temperature, reducing its ability to absorb heat
        efficiently.
      </Para>
      <Para>
        Reducing the width while increasing the height allows the mass flow rate to be kept constant and delays heat
        absorption until the throat, where it is most needed.
      </Para>

    </>
  );
}

function StarshipContent() {
  return (
    <>
      <Para>
        Being a great fan of the <strong className="text-white">Starship</strong> rocket project led by SpaceX, I have
        been closely following its development in detail since 2019. One of the major phases of this development is
        pressure testing of the LOX (liquid oxygen) and methane tanks.
      </Para>
      <Para>
        The goal is to fill the tank with cryogenic liquid nitrogen and increase the internal pressure until rupture,
        in order to analyse the failure mode. Anticipating this test and having access to a tank drawing published
        online by an employee — including dimensions and sheet thickness — I reconstructed the tank geometry and
        performed a hydrostatic pressure simulation to attempt to predict the failure zone and the critical pressure
        using the finite element method.
      </Para>

      <Fig small src={`${BASE}/image51.png`} caption="FEA simulation and failure zone prediction" />

      <Outcome accent="#4CAF50" title="Results">
        <ul className="space-y-1">
          <li><strong className="text-white">Actual tank failure pressure:</strong> 7.6 bar</li>
          <li><strong className="text-white">Predicted failure pressure:</strong> 8.1 bar</li>
          <li><strong className="text-white">Difference:</strong> Likely due to temperature being higher in the actual
            tests (the ultimate strength of stainless steel increases as temperature decreases).</li>
          <li><strong className="text-white">Failure zone:</strong> Predicted by the red zone (maximum stress). In the
            actual test, failure occurred exactly in the zone predicted by the finite element method: at the end of the
            arc and beginning of the linear section.</li>
        </ul>
      </Outcome>
      <Para>
        Watching the test live and accurately predicting the exact failure zone was particularly striking and validated
        the simulation approach used.
      </Para>
    </>
  );
}

function LQRContent() {
  return (
    <>
      <Subsubsection title="Context" />
      <Para>
        A traditional rocket is thrown away after a single flight — the equivalent of scrapping a commercial aircraft
        after every trip. SpaceX changed this by landing their boosters propulsively: using the rocket's own engines
        to decelerate and return it to the launch site, where it can be refuelled and flown again. This is what drives
        their cost advantage. Starship's Super Heavy booster goes further — it is caught mid-air by two mechanical
        arms on the launch tower, since a booster of that size landing on legs would be structurally impractical.
      </Para>
      <Para>
        Making this work is a flight software problem. The booster is an inherently unstable system — it naturally
        tips over. Its engines can be gimballed (their thrust direction tilted), and the flight software must
        coordinate those gimbal angles in real time, at every millisecond, to simultaneously control where the vehicle
        is going and how it is oriented. An uncorrected error compounds faster than a human could react.
      </Para>
      <Para>
        This project builds that flight software from scratch. A 6-DOF rigid-body simulator acts as the virtual
        rocket — it takes engine commands and propagates the full physical state of the vehicle forward in time.
        An LQR full-state feedback controller reads that state, computes the optimal gimbal angles and thrust for each
        engine, and sends the commands back. Closing this loop is called a Software-in-the-Loop (SIL) simulation:
        the standard method for validating flight software before it ever runs on real hardware.
      </Para>

      <div className="grid grid-cols-2 gap-4 my-5">
        <div className="flex flex-col items-center gap-2">
          <div className="rounded-md overflow-hidden">
            <video
              src="/ali-portfolio/images-videos/booster-catch-real.mp4"
              className="h-96 w-auto max-w-full"
              autoPlay
              playsInline
              loop
              muted
            />
          </div>
          <p className="text-xs text-gray-500 italic text-center">Super Heavy booster caught by Mechazilla, 13 Oct 2024</p>
        </div>
        <div className="flex flex-col items-center gap-2">
          <div className="rounded-md overflow-hidden">
            <video
              src="/ali-portfolio/images-videos/booster-catch-sim.mp4"
              className="h-96 w-auto max-w-full"
              autoPlay
              playsInline
              loop
              muted
              ref={el => { if (el) el.playbackRate = 1.1; }}
            />
          </div>
          <p className="text-xs text-gray-500 italic text-center">6-DOF LQR simulation — Go to Setpoint manoeuvre</p>
        </div>
      </div>
      <div className="flex justify-center mt-6 mb-3">
        <p className="text-sm text-gray-400 font-light italic">The simulation runs live in your browser — no install needed.</p>
      </div>
      <div className="flex justify-center mb-8">
        <Link to="/rocketDemo"
          className="inline-flex items-center gap-3 px-10 py-3.5 rounded-lg text-sm font-light tracking-wider transition-all duration-300"
          style={{ border: '1px solid rgba(159,142,109,0.45)', color: '#9F8E6D' }}
          onMouseEnter={e => { e.currentTarget.style.background = '#9F8E6D'; e.currentTarget.style.color = 'white'; }}
          onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = '#9F8E6D'; }}>
          <ExternalLink className="w-4 h-4" />
          <span>Launch Demo</span>
        </Link>
      </div>

      <Subsubsection title="Objective" />
      <Para>
        The goal is to develop and validate, in simulation, the full guidance and control stack for a propulsively
        landing rocket stage. This requires two things built in tandem: a numerically reliable 6-DOF rigid-body
        integrator that serves as the virtual vehicle, and an LQR full-state feedback controller that commands it.
        Neither is useful without the other — the integrator needs a controller to drive it to a meaningful scenario,
        and the controller needs a physics-accurate plant to be tested against. The two are developed and unit-tested
        separately, then connected in a Software-in-the-Loop (SIL) simulation that validates closed-loop behaviour at
        flight-realistic update rates. The three parts below follow that order: first the integrator, then the
        controller, then the integration that ties them together.
      </Para>
      <Para>
        Full implementation:{' '}
        <a href="https://github.com/floaty-bone/rocket-integrator" target="_blank" rel="noreferrer"
          className="text-[#9F8E6D] hover:underline">
          github.com/floaty-bone/rocket-integrator
        </a>
        {' · '}
        <Link to="/rocketDemo" className="inline-flex items-center gap-1 text-[#9F8E6D] hover:underline">
          <ExternalLink className="w-3 h-3" />launch demo
        </Link>
      </Para>

      {/* ── Part 1: the integrator ── */}
      <Subsection title="Part 1 — The 6-DOF Rigid-Body Integrator (RK4)" />
      <Para>
        The integrator makes no concessions to simplicity: no linearisation, no small-angle assumption, no analytical
        shortcut. It propagates the full nonlinear state forward in time using a fourth-order Runge-Kutta scheme at
        sub-millisecond resolution, and must remain numerically stable over the full duration of a landing manoeuvre.
      </Para>

      <Subsubsection title="State Representation" />
      <Para>
        The state of the rigid body is encoded in a 13-element vector:
      </Para>
      <EqBlock>{`\\mathbf{s} = \\bigl[\\,\\underbrace{x,\\,y,\\,z}_{\\text{position}},\\;\\underbrace{q_w,\\,q_x,\\,q_y,\\,q_z}_{\\text{attitude}},\\;\\underbrace{v_x,\\,v_y,\\,v_z}_{\\text{velocity}},\\;\\underbrace{\\omega_x,\\,\\omega_y,\\,\\omega_z}_{\\text{angular velocity}}\\,\\bigr]`}</EqBlock>
      <Para>
        Attitude is represented by a unit quaternion rather than Euler angles. This eliminates gimbal lock entirely and
        keeps the kinematics well-conditioned at every orientation, including when the body undergoes large rotations.
      </Para>

      <Subsubsection title="Equations of Motion" />
      <Para>
        The state derivative <Eq>{'\\dot{\\mathbf{s}}'}</Eq> is assembled from four coupled equations evaluated at
        each integration step.
      </Para>
      <p className="text-sm text-white/80 font-medium mt-3 mb-1">Position kinematics.</p>
      <EqBlock>{`\\dot{\\mathbf{r}} = \\mathbf{v}`}</EqBlock>
      <p className="text-sm text-white/80 font-medium mt-3 mb-1">Quaternion kinematics.</p>
      <Para>The attitude evolves according to:</Para>
      <EqBlock>{`\\dot{\\mathbf{q}} = \\frac{1}{2}\\,\\Omega(\\boldsymbol{\\omega})\\,\\mathbf{q}`}</EqBlock>
      <Para>
        where <Eq>{'\\Omega(\\boldsymbol{\\omega})'}</Eq> is the{' '}
        <Eq>{'4\\times4'}</Eq> skew-symmetric matrix constructed from the body-frame angular velocity:
      </Para>
      <EqBlock>{`\\Omega(\\boldsymbol{\\omega}) =
\\begin{pmatrix}
 0        & -\\omega_x & -\\omega_y & -\\omega_z \\\\
 \\omega_x &  0        &  \\omega_z & -\\omega_y \\\\
 \\omega_y & -\\omega_z &  0        &  \\omega_x \\\\
 \\omega_z &  \\omega_y & -\\omega_x &  0
\\end{pmatrix}`}</EqBlock>
      <p className="text-sm text-white/80 font-medium mt-3 mb-1">Translational dynamics.</p>
      <Para>Newton's second law in the inertial frame:</Para>
      <EqBlock>{`\\dot{\\mathbf{v}} = \\frac{1}{m}\\,\\mathbf{R}(\\mathbf{q})\\,\\mathbf{F}_{\\text{body}} + \\mathbf{g}`}</EqBlock>
      <Para>
        where <Eq>{'\\mathbf{R}(\\mathbf{q})'}</Eq> is the rotation matrix from body frame to inertial frame, derived
        directly from the quaternion:
      </Para>
      <EqBlock>{`\\mathbf{R}(\\mathbf{q}) =
\\begin{pmatrix}
1 - 2(q_y^2+q_z^2)   & 2(q_xq_y - q_wq_z) & 2(q_xq_z + q_wq_y) \\\\
2(q_xq_y + q_wq_z)   & 1 - 2(q_x^2+q_z^2) & 2(q_yq_z - q_wq_x) \\\\
2(q_xq_z - q_wq_y)   & 2(q_yq_z + q_wq_x) & 1 - 2(q_x^2+q_y^2)
\\end{pmatrix}`}</EqBlock>
      <p className="text-sm text-white/80 font-medium mt-3 mb-1">Rotational dynamics.</p>
      <Para>Euler's rotation equation in the body frame:</Para>
      <EqBlock>{`\\dot{\\boldsymbol{\\omega}} = \\mathbf{I}^{-1}\\bigl(\\mathbf{M} - \\boldsymbol{\\omega}\\times(\\mathbf{I}\\,\\boldsymbol{\\omega})\\bigr)`}</EqBlock>
      <Para>
        The term <Eq>{'\\boldsymbol{\\omega}\\times(\\mathbf{I}\\,\\boldsymbol{\\omega})'}</Eq> is the gyroscopic term.
        It accounts for the redistribution of angular momentum between axes whenever the angular velocity vector is not
        aligned with a principal axis of inertia. It is this term that governs all non-trivial rotational behaviour.
      </Para>

      <Subsubsection title="The RK4 Scheme" />
      <Para>
        The four equations above are bundled into a single state-derivative function{' '}
        <Eq>{'f(\\mathbf{s})'}</Eq>. The classical fourth-order Runge-Kutta method advances the state by one
        time-step <Eq>{'h'}</Eq>:
      </Para>
      <EqBlock>{`\\begin{aligned}
\\mathbf{k}_1 &= f(\\mathbf{s}_n) \\\\
\\mathbf{k}_2 &= f\\!\\left(\\mathbf{s}_n + \\tfrac{h}{2}\\,\\mathbf{k}_1\\right) \\\\
\\mathbf{k}_3 &= f\\!\\left(\\mathbf{s}_n + \\tfrac{h}{2}\\,\\mathbf{k}_2\\right) \\\\
\\mathbf{k}_4 &= f\\!\\left(\\mathbf{s}_n + h\\,\\mathbf{k}_3\\right) \\\\[6pt]
\\mathbf{s}_{n+1} &= \\mathbf{s}_n + \\frac{h}{6}\\bigl(\\mathbf{k}_1 + 2\\mathbf{k}_2 + 2\\mathbf{k}_3 + \\mathbf{k}_4\\bigr)
\\end{aligned}`}</EqBlock>
      <Para>
        The local truncation error is <Eq>{'\\mathcal{O}(h^5)'}</Eq>, giving a global error of{' '}
        <Eq>{'\\mathcal{O}(h^4)'}</Eq>. At <Eq>{'h = 1\\,\\text{ms}'}</Eq>, this is more than sufficient for the
        dynamics of interest.
      </Para>
      <Para>
        <strong className="text-white">Numerical reliability.</strong> Two measures preserve long-term accuracy. The
        inverse inertia tensor <Eq>{'\\mathbf{I}^{-1}'}</Eq> is computed once at initialisation and cached, avoiding
        repeated matrix factorisation at every step. The quaternion is re-normalised after every step to prevent slow
        numerical drift from corrupting the attitude representation:
      </Para>
      <EqBlock>{`\\mathbf{q} \\leftarrow \\frac{\\mathbf{q}}{\\|\\mathbf{q}\\|}`}</EqBlock>

      <Subsubsection title="A Fun Observation: the Intermediate Axis Theorem" />
      <Para>
        Once the integrator was running, the inertia tensor was set to three distinct principal moments,{' '}
        <Eq>{'I_1 = 300'}</Eq>, <Eq>{'I_2 = 100'}</Eq>, <Eq>{'I_3 = 30\\,\\text{kg}\\cdot\\text{m}^2'}</Eq>, and the
        body was given an initial angular velocity nearly aligned with the intermediate axis, with a small perturbation
        on the other two. The gyroscopic term in Euler's equation then drives the system into an unstable regime: the
        Dzhanibekov effect. The body undergoes periodic half-turns and re-aligns, indefinitely. The real-time 3D
        animation made this directly visible, which is something no closed-form solution could have offered as clearly.
      </Para>

      {/* ── Part 2: the controller ── */}
      <Subsection title="Part 2 — The LQR Controller" />
      <Para>
        An LQR (Linear Quadratic Regulator) is a full-state feedback controller: it multiplies the current state error
        by a precomputed gain matrix <Eq>{'\\mathbf{K}'}</Eq> to produce the control command that optimally balances how
        quickly the error is driven to zero against how much actuator effort is spent. The gain is computed from a
        tangent-space linearisation of the 6-DOF dynamics, and is what turns the raw simulator into a controllable
        vehicle.
      </Para>

      <Subsubsection title="Linearisation and Gain Computation" />
      <Para>
        LQR is, by definition, a linear law: it is exact only at the operating point{' '}
        <Eq>{'(\\mathbf{s}_{\\text{op}},\\mathbf{u}_{\\text{op}})'}</Eq> where the Jacobians{' '}
        <Eq>{'A=\\partial f/\\partial\\mathbf{s}'}</Eq> and{' '}
        <Eq>{'B=\\partial f/\\partial\\mathbf{u}'}</Eq> were taken. The vehicle dynamics are strongly non-linear
        (<Eq>{'\\mathbf{R}(\\mathbf{q})'}</Eq>, gyroscopic coupling,{' '}
        <Eq>{'\\boldsymbol{\\omega}\\times(\\mathbf{I}\\boldsymbol{\\omega})'}</Eq>), so the operating point must be
        refreshed along the trajectory. At each refresh, <Eq>{'A'}</Eq> and <Eq>{'B'}</Eq> are recomputed by JAX
        auto-differentiation of the dynamics function around the current <Eq>{'(\\mathbf{s},\\mathbf{u})'}</Eq>, and a
        new gain <Eq>{'\\mathbf{K}'}</Eq> is obtained by solving the continuous-time algebraic Riccati equation:
      </Para>
      <EqBlock>{`A^\\top P + P A - P B R^{-1} B^\\top P + Q = 0,
\\qquad
\\mathbf{K} = R^{-1} B^\\top P`}</EqBlock>

      <Subsubsection title="Thrust-Vector-Control Encoding" />
      <Para>
        Internally the LQR computes thrust as a Cartesian force vector{' '}
        <Eq>{'[F_x, F_y, F_z]'}</Eq> per engine, which is the natural form for an affine state-space formulation. The
        real vehicle, however, commands two gimbal angles and a thrust magnitude per engine. The controller therefore
        converts its Cartesian solution into the <Eq>{'[\\alpha,\\,\\beta,\\,T]'}</Eq> form an actuator actually
        receives, via the forward and inverse trigonometric mapping:
      </Para>
      <EqBlock>{`\\begin{aligned}
F_x &= T\\cos\\alpha\\cos\\beta \\\\
F_y &= T\\cos\\alpha\\sin\\beta \\\\
F_z &= -T\\sin\\alpha
\\end{aligned}
\\qquad
\\begin{aligned}
T     &= \\sqrt{F_x^2+F_y^2+F_z^2} \\\\
\\alpha &= -\\arcsin\\!\\bigl(F_z/T\\bigr) \\\\
\\beta  &= \\operatorname{atan2}(F_y,\\,F_x)
\\end{aligned}`}</EqBlock>

      {/* ── Part 3: SIL integration & results ── */}
      <Subsection title="Part 3 — Software-in-the-Loop Integration & Results" />
      <Para>
        The integrator and controller are developed and unit-tested in isolation; neither result says anything about
        closed-loop behaviour. The Software-in-the-Loop (SIL) simulation closes the loop: at every controller tick,
        the gain <Eq>{'\\mathbf{K}'}</Eq> is applied to the live plant state, the resulting actuator command is fed
        back into the RK4 integrator, and the new state is fed back into the controller. The purpose is to validate
        that the LQR law, re-linearised periodically along the trajectory, actually stabilises the full non-linear
        vehicle from a non-trivial initial attitude to a target setpoint.
      </Para>

      <Subsubsection title="Architecture" />
      <Para>
        The SIL is built as two decoupled blocks exchanging a strictly defined interface. This boundary is the SIL
        contract: the controller block knows nothing about the integrator, the plant block knows nothing about the gain
        matrix, and the bus between them carries only signals that would exist on the real vehicle.
      </Para>
      <Para><strong className="text-white">Controller block.</strong></Para>
      <ul className="text-sm text-gray-300 space-y-1 mb-3 pl-4">
        <li><strong className="text-white">Input:</strong> current state <Eq>{'\\mathbf{s}\\in\\mathbb{R}^{13}'}</Eq>, target setpoint <Eq>{'\\mathbf{s}^\\star\\in\\mathbb{R}^{13}'}</Eq>.</li>
        <li><strong className="text-white">Output:</strong> TVC command <Eq>{'\\mathbf{u}_{\\text{gimbal}}\\in\\mathbb{R}^{9}'}</Eq>, encoded as <Eq>{'[\\alpha,\\,\\beta,\\,T]'}</Eq> per engine.</li>
      </ul>
      <Para><strong className="text-white">Plant block.</strong></Para>
      <ul className="text-sm text-gray-300 space-y-1 mb-3 pl-4">
        <li><strong className="text-white">Input:</strong> current state <Eq>{'\\mathbf{s}\\in\\mathbb{R}^{13}'}</Eq>, TVC command <Eq>{'\\mathbf{u}_{\\text{gimbal}}\\in\\mathbb{R}^{9}'}</Eq>.</li>
        <li><strong className="text-white">Output:</strong> next state <Eq>{'\\mathbf{s}^+\\in\\mathbb{R}^{13}'}</Eq>, computed by one RK4 step.</li>
      </ul>
      <Para>
        The crucial design choice is the bus variable. Although the controller solves internally in Cartesian forces,
        the bus carries the <Eq>{'[\\alpha,\\,\\beta,\\,T]'}</Eq> gimbal command — the same signal the real vehicle
        receives. Routing it across the SIL boundary forces the simulation to exercise the same trigonometric mapping
        shown above, so any singularity, saturation, or precision loss in that mapping appears in closed loop, not as
        a hidden internal quantity.
      </Para>

      <Subsubsection title="Multi-Rate Scheduling" />
      <Para>
        A real flight stack does not run every block at the same frequency, and the SIL reproduces that. Three
        independent rates are scheduled inside a single loop:
      </Para>
      <DataTable
        caption="Scheduled rates inside the SIL loop"
        head={['Block', 'Rate', 'Justification']}
        rows={[
          ['RK4 integration', '8000 Hz', 'h = 0.125 ms, global error O(h⁴)'],
          ['Controller update', '5000 Hz', 'Realistic upper bound for an embedded LQR'],
          ['LQR re-linearisation', '30 Hz', 'JAX auto-diff of f(s,u) and Riccati solve'],
        ]}
      />
      <Para>
        The relinearisation rate is the load-bearing parameter. Because the LQR gain is only valid near its operating
        point and the dynamics are strongly non-linear, the linearisation must be refreshed along the trajectory —
        here at 30 Hz. Between refreshes, the controller applies the cached <Eq>{'\\mathbf{K}'}</Eq> at the full 5 kHz
        rate; the cost stays in the affordable range while the linearisation stays close enough to the trajectory to
        remain valid.
      </Para>

      <Subsubsection title="Test Scenario" />
      <Para>
        The vehicle is a 120-tonne stage with three gimballed engines clustered at radius{' '}
        <Eq>{'a = 1.5\\,\\text{m}'}</Eq>, mounted <Eq>{'l = 18\\,\\text{m}'}</Eq> below the centre of mass, with
        inertia{' '}
        <Eq>{'\\operatorname{diag}(1.2\\times10^6,\\,3.5\\times10^7,\\,3.5\\times10^7)\\,\\text{kg}\\cdot\\text{m}^2'}</Eq>.
        Initial attitude is horizontal <Eq>{'(\\theta = -\\pi/2)'}</Eq> at the origin, with zero linear and angular
        velocity. The setpoint asks for a translation to{' '}
        <Eq>{'(50,\\,100,\\,60)\\,\\text{m}'}</Eq> while holding the same horizontal attitude. The nominal thrust per
        engine is the static hover share, <Eq>{'mg/3'}</Eq>.
      </Para>
      <Para>
        The LQR weights penalise position error heavily (<Eq>{'Q_{\\text{pos}} = 10^7'}</Eq>) and linear velocity
        moderately (<Eq>{'Q_v = 10^5'}</Eq>); attitude and angular rate weights are kept low enough to let the
        controller redistribute thrust freely. The control cost <Eq>{'R = I_9'}</Eq> is uniform across the nine
        actuator channels.
      </Para>

      <Subsubsection title="Results" />
      <Para>
        The closed loop converges. The vehicle drives all three position channels onto their setpoint with no
        steady-state offset and no oscillation past the transient. The TVC commands stay inside physically sensible
        bounds: gimbal angles in the low-degree range, per-engine thrust modulating around the hover share. The final
        quaternion norm, after 30 seconds at 8 kHz (<Eq>{'2.4 \\times 10^5'}</Eq> RK4 steps), is within{' '}
        <Eq>{'10^{-8}'}</Eq> of unity, confirming that the per-step re-normalisation of the integrator holds up over
        the full SIL horizon.
      </Para>
      <Para>
        The most useful diagnostic is the disagreement curve between the controller's internal Cartesian command and
        the gimbal-encoded command actually delivered to the plant. The two are mathematically inverse on paper; in the
        SIL they round-trip through floating point, so any divergence flags either a singularity in the{' '}
        <Eq>{'[\\alpha,\\beta,T]'}</Eq> chart or a saturation in the controller. Across the test scenario the
        disagreement stays at machine precision, which is the strongest possible statement that the SIL boundary is
        exercised cleanly and the controller never asks for a command the real vehicle could not execute.
      </Para>

      <Outcome accent="#9F8E6D" title="Key Outcomes">
        <ul className="space-y-1">
          <li>· Full 6-DOF rigid-body propagation with quaternion kinematics and no gimbal lock; fourth-order Runge-Kutta at <Eq>{'h = 1\\,\\text{ms}'}</Eq>, global error <Eq>{'\\mathcal{O}(h^4)'}</Eq>, with per-step quaternion re-normalisation against long-term drift.</li>
          <li>· Numerical reproduction of the Dzhanibekov effect: rotation about the intermediate inertia axis is unstable and the simulator captures it exactly.</li>
          <li>· LQR re-linearised on-trajectory via JAX auto-diff + Riccati solve, no hand-derived Jacobians; SIL boundary carries <Eq>{'[\\alpha,\\beta,T]'}</Eq>, the real TVC command, not internal Cartesian forces.</li>
          <li>· Closed-loop validation across the full SIL boundary at flight-realistic rates (5 kHz control, 8 kHz plant, 30 Hz re-linearisation), converging from a horizontal initial attitude to a 3-axis position setpoint with quaternion norm preserved to <Eq>{'10^{-8}'}</Eq> after <Eq>{'2.4\\times10^5'}</Eq> integration steps.</li>
        </ul>
      </Outcome>
    </>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

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
    setTimeout(() => window.scrollTo({ top: document.documentElement.scrollHeight, behavior: 'smooth' }), 100);
  };

  return (
    <div className="min-h-screen text-white" style={{ background: 'transparent' }}>
      <div className="fixed inset-0 z-0">
        <RippleMesh className="w-full h-full" />
      </div>
      <div className="fixed inset-0 z-0" style={{ background: 'rgba(6,6,10,0.75)' }} />

      <nav className={`fixed w-full z-50 transition-all duration-500 ${scrollPosition > 50 ? 'bg-black/90 backdrop-blur-sm' : 'bg-transparent'} ${navVisible ? 'translate-y-0' : '-translate-y-full'}`}>
        <div className="w-full px-24 py-6 flex justify-between items-center">
          <h1 className="text-xl font-light tracking-wide">Ali Abouelazz</h1>
          <div className="flex gap-12 text-sm font-light tracking-wider">
            <Link to="/home" className="hover:text-[#9F8E6D] transition-colors duration-300">HOME</Link>
            <Link to="/downloadsPage" className="hover:text-[#9F8E6D] transition-colors duration-300">TECHNICAL PORTFOLIO</Link>
            <Link to="/competencesPage" className="hover:text-[#9F8E6D] transition-colors duration-300">SKILLS</Link>
            <Link to="/loisirs" className="hover:text-[#9F8E6D] transition-colors duration-300">INTERESTS</Link>
            <Link to="/rocketDemo" className="hover:text-[#9F8E6D] transition-colors duration-300">ROCKET DEMO</Link>
            <a href="#" onClick={handleContactClick} className="hover:text-[#9F8E6D] transition-colors duration-300">CONTACT</a>
          </div>
        </div>
      </nav>

      <div className="relative z-10 max-w-4xl mx-auto px-8 pt-40 pb-20">
        <div className="mb-16">
          <h2 className="text-5xl font-light text-white mb-4">Technical Portfolio</h2>
          <p className="text-gray-400 font-light max-w-xl leading-relaxed">
            Internship projects and personal studies in numerical simulation, structural analysis, and flight control.
          </p>
          <a
            href="/ali-portfolio/downloads/Portfolio-technique.pdf"
            download
            className="inline-flex items-center gap-2 mt-6 text-xs font-light tracking-wider text-gray-500 hover:text-[#9F8E6D] transition-colors duration-300 border border-white/10 hover:border-[#9F8E6D]/30 rounded-md px-4 py-2"
          >
            <Download className="w-3.5 h-3.5" />
            <span>DOWNLOAD PDF VERSION</span>
          </a>
        </div>

        <div className="mb-4">
          <p className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-4">Featured</p>
          <RocketDemoCard />
        </div>

        <div className="mt-12 space-y-4">
          <MajorSection title="Personal Projects">
            <SubSection label="Personal Study" title="LQR Full-State Feedback Control & 6-DOF Rocket Integrator" accent="#9F8E6D">
              <LQRContent />
            </SubSection>
            <SubSection label="Personal Study" title="Failure Prediction of the Starship Tank" accent="#9F8E6D">
              <StarshipContent />
            </SubSection>
            <SubSection label="Personal Study" title="Optimisation of Regenerative Cooling in a Combustion Chamber" accent="#9F8E6D">
              <CombustionContent />
            </SubSection>
          </MajorSection>

          <MajorSection title="Internships">
            <SubSection label="Final Year Internship" title="Caterpillar" accent="#E8A020">
              <CatContent />
            </SubSection>
            <SubSection label="Engineering Internship" title="General Electric Vernova" accent="#4A9EBB">
              <GEContent />
            </SubSection>
          </MajorSection>
        </div>
      </div>

      <section className="relative z-10 py-40 px-24 mt-10">
        <div className="max-w-7xl mx-auto flex justify-between items-start">
          <div className="max-w-lg">
            <h3 className="text-4xl font-light mb-8">Contact Me</h3>
            <div className="space-y-8">
              <div>
                <span className="text-xs font-semibold text-[#9F8E6D] uppercase tracking-widest mb-2 block">Email</span>
                <a href="mailto:ali.abouelazz@gmail.com" className="text-lg font-light text-white hover:text-[#9F8E6D] transition-colors duration-300">
                  ali.abouelazz@gmail.com
                </a>
              </div>
              <div>
                <span className="text-xs font-semibold text-[#9F8E6D] uppercase tracking-widest mb-2 block">WhatsApp contact only</span>
                <a href="tel:+33777451629" className="text-lg font-light text-white hover:text-[#9F8E6D] transition-colors duration-300">
                  +33 7 77 45 16 29
                </a>
              </div>
            </div>
          </div>
          <div className="w-[200px] flex justify-center items-center">
            <img src="/ali-portfolio/images-videos/profilePic.png" alt="Mohamed Ali Abouelazz Profile"
              className="w-full h-auto object-cover rounded-lg shadow-lg" />
          </div>
        </div>
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
