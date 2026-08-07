import { describe, expect, it } from "vitest";
import { getGuide } from "./guide";
import { languageOptions, translations, type Language } from "./i18n";

const languages = languageOptions.map((item) => item.code);

describe("language modules", () => {
  it("keeps every language aligned with the English translation schema", () => {
    const topLevelKeys = Object.keys(translations.en).sort();
    const uiKeys = Object.keys(translations.en.ui).sort();
    const promptCraftKeys = Object.keys(translations.en.ui.promptCraftHints).sort();

    for (const language of languages) {
      expect(Object.keys(translations[language]).sort(), language).toEqual(topLevelKeys);
      expect(Object.keys(translations[language].ui).sort(), `${language}.ui`).toEqual(uiKeys);
      expect(
        Object.keys(translations[language].ui.promptCraftHints).sort(),
        `${language}.ui.promptCraftHints`
      ).toEqual(promptCraftKeys);
    }
  });

  it("provides a real How to Use guide in every selectable language", () => {
    for (const language of languages as Language[]) {
      const guide = getGuide(language);
      expect(guide.length, language).toBeGreaterThan(1);
      expect(guide.flatMap((section) => section.entries).length, language).toBeGreaterThanOrEqual(8);
    }
  });
});
