import type { RefObject } from "react";
import { Play, ShieldAlert, SlidersHorizontal, Square, Trash2 } from "lucide-react";
import { jailbreakModeLabel } from "../../app/benchmarking";
import type { Translation } from "../../i18n";
import type { ChatTurn, JailbreakMode, OutputAssessment } from "../../types";

interface ChatTabProps {
  t: Translation;
  messages: ChatTurn[];
  pendingUserMessage: string | null;
  status: "idle" | "running" | "done" | "error";
  generatedText: string;
  emptyOutputNotice: boolean;
  prompt: string;
  onPromptChange: (value: string) => void;
  onStart: () => void;
  onStop: () => void;
  onOpenSettings: () => void;
  threadRef: RefObject<HTMLDivElement | null>;
  modelLabel: string;
  jailbreak: boolean;
  jailbreakMode: JailbreakMode;
  onJailbreakChange: (value: boolean) => void;
  isApiAdapter: boolean;
  activeRuleCount: number;
  outputTokens: number | null;
  effectiveMaxTokens: number | null;
  busy: boolean;
  outputAssessment: OutputAssessment | null;
}

export function ChatTab({
  t, messages, pendingUserMessage, status, generatedText, emptyOutputNotice,
  prompt, onPromptChange, onStart, onStop, onOpenSettings, threadRef, modelLabel,
  jailbreak, jailbreakMode, onJailbreakChange, isApiAdapter, activeRuleCount,
  outputTokens, effectiveMaxTokens, busy, outputAssessment
}: ChatTabProps) {
  return (
    <div className="chat-page">
      <div className="thread" ref={threadRef}>
        <div className="thread-inner">
          {messages.length === 0 && !pendingUserMessage && status !== "running" ? (
            <div className="thread-empty">
              <h2>{t.chatEmptyTitle}</h2>
              <p>{t.chatEmptyHint}</p>
              <div className="starter-grid">
                {t.chatStarters.map((starter) => <button key={starter} className="starter" onClick={() => onPromptChange(starter)}>{starter}</button>)}
              </div>
            </div>
          ) : null}

          {messages.map((message, index) => (
            <article className={`msg ${message.role}`} key={`msg-${index}`}>
              <div className="msg-role">{message.role === "user" ? t.chatYou : t.chatModel}</div>
              <div className="msg-body">{message.content}</div>
            </article>
          ))}
          {pendingUserMessage ? <article className="msg user"><div className="msg-role">{t.chatYou}</div><div className="msg-body">{pendingUserMessage}</div></article> : null}
          {status === "running" ? <article className="msg assistant"><div className="msg-role">{t.chatModel}</div><div className="msg-body">{generatedText || <span className="caret" />}</div></article> : null}
          {status !== "running" && emptyOutputNotice ? <article className="msg assistant"><div className="msg-role">{t.chatModel}</div><div className="msg-body">{t.ui.emptyOutput}</div></article> : null}
        </div>
      </div>

      <div className="composer-wrap">
        <div className="composer">
          <textarea
            className="composer-input"
            placeholder={t.prompt}
            value={prompt}
            onChange={(event) => onPromptChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                if (status !== "running") onStart();
              }
            }}
            rows={1}
            spellCheck={false}
          />
          <div className="composer-bar">
            <div className="composer-chips">
              <button className="chip" onClick={onOpenSettings} title={t.navSettings}><SlidersHorizontal size={13} />{modelLabel}</button>
              <button className={`chip${jailbreak ? " on" : ""}`} onClick={() => onJailbreakChange(!jailbreak)} title={t.jailbreakHint} disabled={isApiAdapter}><ShieldAlert size={13} />{jailbreak ? jailbreakModeLabel(jailbreakMode, t) : t.jailbreakOff}</button>
              {activeRuleCount > 0 ? <button className="chip on" onClick={onOpenSettings} title={`${activeRuleCount} ${t.activeRules}`}><Trash2 size={13} />{activeRuleCount}</button> : null}
            </div>
            <div className="composer-send">
              {outputTokens !== null ? <span className="token-hint">{outputTokens}{effectiveMaxTokens !== null ? ` / ${effectiveMaxTokens}` : ""} {t.outputTokens}</span> : null}
              {status === "running" ? <button className="primary" onClick={onStop} title={t.stopRunTitle}><Square size={15} /> {t.stop}</button> : <button className="primary" onClick={onStart} disabled={busy || !prompt.trim()} title={t.startRunTitle}><Play size={15} /> {t.run}</button>}
            </div>
          </div>
        </div>
        {isApiAdapter ? <p className="composer-note">{t.apiAdapterWarning}</p> : null}
        {outputAssessment ? <div className={`unsupported-banner${outputAssessment.manual_review_required ? " warn-note" : ""}`}>{t.ui.outcome}: <strong>{outputAssessment.category}</strong> · {outputAssessment.complete ? t.ui.complete : t.ui.incomplete} · {outputAssessment.coherent ? t.ui.coherent : t.ui.incoherent}{outputAssessment.manual_review_required ? ` · ${t.ui.manualReviewRequired}` : ""}</div> : null}
      </div>
    </div>
  );
}
