export interface GuideEntry {
  term: string;
  body: string;
}

export interface GuideSection {
  title: string;
  intro?: string;
  entries: GuideEntry[];
}
