import { useRef, useState } from "react";
import { getGuide } from "../../guide";
import type { Language } from "../../i18n";

export function GuideTab({ language }: { language: Language }) {
  const sections = getGuide(language);
  const [activeTab, setActiveTab] = useState(0);
  const bodyRef = useRef<HTMLDivElement>(null);
  const sectionRefs = useRef<(HTMLElement | null)[]>([]);

  const slugify = (title: string) =>
    title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");

  const handleTabClick = (index: number) => {
    setActiveTab(index);
    const section = sectionRefs.current[index];
    if (section && bodyRef.current) {
      bodyRef.current.scrollTo({ top: section.offsetTop - 12, behavior: "smooth" });
    }
  };

  const handleScroll = () => {
    if (!bodyRef.current) return;
    const scrollTop = bodyRef.current.scrollTop;
    let best = 0;
    sectionRefs.current.forEach((section, index) => {
      if (section && section.offsetTop - 40 <= scrollTop) best = index;
    });
    setActiveTab(best);
  };

  return (
    <div className="guide-tab">
      <nav className="guide-tabs-nav" role="tablist">
        {sections.map((section, index) => (
          <button
            key={section.title}
            role="tab"
            aria-selected={activeTab === index}
            className={`guide-tab-btn${activeTab === index ? " active" : ""}`}
            onClick={() => handleTabClick(index)}
            title={section.title}
          >
            {section.title}
          </button>
        ))}
      </nav>
      <div className="guide-body" ref={bodyRef} onScroll={handleScroll}>
        {sections.map((section, index) => (
          <section
            className="guide-section"
            key={section.title}
            id={slugify(section.title)}
            ref={(element) => { sectionRefs.current[index] = element; }}
          >
            <h3>{section.title}</h3>
            {section.intro ? <p className="guide-intro">{section.intro}</p> : null}
            {section.entries.length ? (
              <div className="guide-entries">
                {section.entries.map((entry) => (
                  <article key={entry.term}>
                    <strong>{entry.term}</strong>
                    <span style={{ whiteSpace: "pre-wrap" }}>{entry.body}</span>
                  </article>
                ))}
              </div>
            ) : null}
          </section>
        ))}
      </div>
    </div>
  );
}
