import { translation as de } from "./languages/de";
import { translation as en } from "./languages/en";
import { translation as es } from "./languages/es";
import { translation as tr } from "./languages/tr";

export type Language = "tr" | "en" | "de" | "es";

export const languageOptions: Array<{ code: Language; label: string }> = [
  { code: "tr", label: "Türkçe" },
  { code: "en", label: "English" },
  { code: "de", label: "Deutsch" },
  { code: "es", label: "Español" }
];

export const translations = { tr, en, de, es } as const;

export type Translation = (typeof translations)[Language];

export function conceptLabel(language: Language, concept: string): string {
  const local = translations[language].conceptsMap as Record<string, string>;
  const fallback = translations.en.conceptsMap as Record<string, string>;
  return local[concept] ?? fallback[concept] ?? concept;
}

export function safetyStateLabel(language: Language, state: string): string {
  const local = translations[language].states as Record<string, string>;
  return local[state] ?? state;
}

export function safetyNote(language: Language, safetyState: string, backendNote: string): string {
  const local = translations[language].safetyNotes as Record<string, string>;
  const english = translations.en.safetyNotes as Record<string, string>;
  return local[safetyState] ?? english[safetyState] ?? backendNote;
}
