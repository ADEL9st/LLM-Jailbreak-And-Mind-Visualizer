import { guide as de } from "./languages/de";
import { guide as en } from "./languages/en";
import { guide as es } from "./languages/es";
import { guide as tr } from "./languages/tr";
import type { Language } from "./i18n";
import type { GuideSection } from "./languages/types";

export type { GuideEntry, GuideSection } from "./languages/types";

const guides: Record<Language, GuideSection[]> = { tr, en, de, es };

export function getGuide(language: Language): GuideSection[] {
  return guides[language];
}
