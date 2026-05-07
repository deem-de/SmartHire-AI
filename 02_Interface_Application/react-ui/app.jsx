const { useEffect, useMemo, useState } = React;

const NAV_ITEMS = [
  { id: "resume", label: "Resume", icon: "resume" },
  { id: "interview", label: "Interview", icon: "interview" },
  { id: "settings", label: "Settings", icon: "settings" },
];

const EMPTY_PROFILE = {
  name: "Waiting for Resume",
  title: "Resume Candidate",
  roleMatch: 0,
  location: "Not detected",
  email: "Not detected",
  phone: "Not detected",
  skills: [],
  education: "Not detected",
  major: "Not detected",
  university: "Not detected",
  graduation: "Not detected",
  company: "Not detected",
  experienceLength: "Not detected",
  experienceHighlights: [],
  languages: [],
  projects: [],
  summary: "Upload a resume to generate a structured BERT NER view.",
};

const EMPTY_STATS = [
  { label: "Skill", value: 0, color: "bg-brand-500" },
  { label: "Education", value: 0, color: "bg-emerald-500" },
  { label: "Experience", value: 0, color: "bg-orange-400" },
  { label: "Contact", value: 0, color: "bg-violet-500" },
];

const EMPTY_METRICS = [
  { title: "Baseline Model", value: "--", note: "Confidence", color: "text-brand-500" },
  { title: "Section-Aware Model", value: "--", note: "Confidence", color: "text-emerald-500" },
  { title: "Best Confidence", value: "--", note: "Process a resume first", color: "text-brand-500" },
  { title: "Best Model", value: "Waiting", note: "Compare models on this resume", color: "text-emerald-500", smaller: true },
];

function cx(...classes) {
  return classes.filter(Boolean).join(" ");
}

function nowLabel() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

async function readJsonResponse(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || payload.message || "Request failed.");
  }
  return payload;
}

function App() {
  const [activeSection, setActiveSection] = useState("resume");
  const [language, setLanguage] = useState("English");
  const [selectedModel, setSelectedModel] = useState("BERT Only");
  const [theme, setTheme] = useState("light");
  const [autoSave, setAutoSave] = useState(true);
  const [highlightEntities, setHighlightEntities] = useState(true);

  const [systemInfo, setSystemInfo] = useState({
    bestParserModel: "section_aware",
    aiAvailable: false,
    device: "loading",
  });

  const [uploadedFile, setUploadedFile] = useState(null);
  const [bootstrapping, setBootstrapping] = useState(true);
  const [processing, setProcessing] = useState(false);
  const [processed, setProcessed] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [noticeMessage, setNoticeMessage] = useState("");

  const [extractedText, setExtractedText] = useState(
    "Upload a resume file, then press Process Document to run the trained BERT parser."
  );
  const [profileRaw, setProfileRaw] = useState({});
  const [profile, setProfile] = useState(EMPTY_PROFILE);
  const [stats, setStats] = useState(EMPTY_STATS);
  const [metrics, setMetrics] = useState(EMPTY_METRICS);
  const [parserLabel, setParserLabel] = useState("Section-Aware BERT");

  const [questions, setQuestions] = useState([]);
  const [chatMessages, setChatMessages] = useState([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answerInput, setAnswerInput] = useState("");
  const [feedback, setFeedback] = useState(null);
  const [interviewResults, setInterviewResults] = useState([]);
  const [summary, setSummary] = useState("");
  const [interviewStarting, setInterviewStarting] = useState(false);
  const [answerSubmitting, setAnswerSubmitting] = useState(false);
  const [awaitingNextQuestion, setAwaitingNextQuestion] = useState(false);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  useEffect(() => {
    let active = true;

    async function bootstrap() {
      try {
        const response = await fetch("/api/bootstrap");
        const data = await readJsonResponse(response);
        if (!active) return;

        setSystemInfo({
          bestParserModel: data.bestParserModel || "section_aware",
          aiAvailable: Boolean(data.aiAvailable),
          device: data.device || "cpu",
        });
        if (!data.aiAvailable) {
          setSelectedModel("BERT Only");
        }
        setParserLabel(data.bestParserModel === "baseline" ? "Baseline BERT" : "Section-Aware BERT");
      } catch (error) {
        if (!active) return;
        setNoticeMessage(`Bootstrap warning: ${error.message}`);
      } finally {
        if (active) setBootstrapping(false);
      }
    }

    bootstrap();
    return () => {
      active = false;
    };
  }, []);

  const initials = useMemo(() => {
    const baseName = profile.name || "CA";
    return baseName
      .split(" ")
      .slice(0, 2)
      .map((part) => part[0] || "")
      .join("")
      .toUpperCase();
  }, [profile.name]);

  const currentQuestion = questions[currentQuestionIndex] || null;

  function resetInterviewState() {
    setQuestions([]);
    setChatMessages([]);
    setCurrentQuestionIndex(0);
    setAnswerInput("");
    setFeedback(null);
    setInterviewResults([]);
    setSummary("");
    setAwaitingNextQuestion(false);
  }

  async function handleFileChange(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploadedFile(file);
    setProcessed(false);
    setErrorMessage("");
    setNoticeMessage("");
    setMetrics(EMPTY_METRICS);
    setExtractedText(`${file.name}\n\nFile selected. Press Process Document to run extraction and parsing.`);
    resetInterviewState();
  }

  async function handleProcess() {
    if (!uploadedFile) {
      setErrorMessage("Please upload a PDF, DOCX, or TXT resume first.");
      return;
    }

    setProcessing(true);
    setProcessed(false);
    setErrorMessage("");
    setNoticeMessage("");
    resetInterviewState();

    try {
      const formData = new FormData();
      formData.append("file", uploadedFile);
      formData.append("parser_model", systemInfo.bestParserModel || "section_aware");

      const response = await fetch("/api/process-resume", {
        method: "POST",
        body: formData,
      });
      const data = await readJsonResponse(response);

      setExtractedText(data.extractedText || "");
      setProfileRaw(data.profileRaw || {});
      setProfile(data.profile || EMPTY_PROFILE);
      setStats(data.stats || EMPTY_STATS);
      setMetrics(data.metrics || EMPTY_METRICS);
      setParserLabel(data.parserLabel || "Section-Aware BERT");
      setProcessed(true);

      if (activeSection === "interview") {
        setChatMessages([
          {
            id: "intro",
            role: "ai",
            type: "message",
            content: "I extracted the candidate profile from the document. Ready to start the interview?",
            time: nowLabel(),
          },
        ]);
      }
    } catch (error) {
      setErrorMessage(error.message);
    } finally {
      setProcessing(false);
    }
  }

  async function startInterview() {
    if (!processed || !Object.keys(profileRaw).length) {
      setErrorMessage("Process a resume first, then start the interview.");
      return;
    }

    setInterviewStarting(true);
    setErrorMessage("");
    setNoticeMessage("");
    setFeedback(null);
    setSummary("");

    try {
      const response = await fetch("/api/start-interview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          profile: profileRaw,
          language,
          selected_model: selectedModel,
          question_count: 3,
        }),
      });
      const data = await readJsonResponse(response);

      const generatedQuestions = data.questions || [];
      setQuestions(generatedQuestions);
      setCurrentQuestionIndex(0);
      setInterviewResults([]);

      if (data.source === "offline" || data.source === "offline_fallback") {
        setNoticeMessage("");
      }

      const introMessages = [
        {
          id: "intro",
          role: "ai",
          type: "message",
          content: "I extracted the candidate profile from the document. Ready to start the interview?",
          time: nowLabel(),
        },
        {
          id: "start",
          role: "user",
          type: "message",
          content: "Start interview",
          time: nowLabel(),
        },
      ];

      if (generatedQuestions[0]) {
        introMessages.push({
          id: "q-0",
          role: "ai",
          type: "question",
          question: generatedQuestions[0],
          time: nowLabel(),
        });
      }

      setChatMessages(introMessages);
      setAwaitingNextQuestion(false);
    } catch (error) {
      setErrorMessage(error.message);
    } finally {
      setInterviewStarting(false);
    }
  }

  async function submitAnswer() {
    const answer = answerInput.trim();
    if (!answer || !currentQuestion) return;

    setAnswerSubmitting(true);
    setErrorMessage("");

    const answerMessage = {
      id: `a-${Date.now()}`,
      role: "user",
      type: "message",
      content: answer,
      time: nowLabel(),
    };
    setChatMessages((prev) => [...prev, answerMessage]);
    setAnswerInput("");

    try {
      const response = await fetch("/api/evaluate-answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: currentQuestion,
          answer,
          profile: profileRaw,
          language,
          selected_model: selectedModel,
        }),
      });
      const data = await readJsonResponse(response);
      const nextFeedback = data.feedback;

      if (data.source === "offline" || data.source === "offline_fallback") {
        setNoticeMessage("");
      }

      setFeedback(nextFeedback);

      const updatedResults = [
        ...interviewResults,
        {
          question: currentQuestion,
          answer,
          feedback: { score_out_of_5: nextFeedback.score, ...nextFeedback },
        },
      ];
      setInterviewResults(updatedResults);

      if (currentQuestionIndex < questions.length - 1) {
        setAwaitingNextQuestion(true);
      } else {
        const summaryResponse = await fetch("/api/final-summary", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            results: updatedResults,
            language,
            selected_model: selectedModel,
          }),
        });
        const summaryData = await readJsonResponse(summaryResponse);
        setSummary(summaryData.summary || "");
        if (summaryData.source === "offline" || summaryData.source === "offline_fallback") {
          setNoticeMessage("");
        }
        setAwaitingNextQuestion(false);
      }
    } catch (error) {
      setErrorMessage(error.message);
    } finally {
      setAnswerSubmitting(false);
    }
  }

  function goToNextQuestion() {
    if (currentQuestionIndex >= questions.length - 1) return;
    const nextIndex = currentQuestionIndex + 1;
    setCurrentQuestionIndex(nextIndex);
    setFeedback(null);
    setAwaitingNextQuestion(false);
    setChatMessages((prev) => [
      ...prev,
      {
        id: `q-${nextIndex}`,
        role: "ai",
        type: "question",
        question: questions[nextIndex],
        time: nowLabel(),
      },
    ]);
  }

  return (
    <div className="min-h-screen grid-backdrop px-5 py-6 text-slateish-700 dark:bg-slateish-900 dark:text-slate-100">
      <div className="mx-auto max-w-[1620px] overflow-hidden rounded-[26px] border border-brand-100 bg-[#f3f4f6] p-4 shadow-soft dark:border-slateish-700 dark:bg-slateish-900">
        <div className="flex min-h-[900px] gap-4">
          <Sidebar activeSection={activeSection} onSelect={setActiveSection} />

          <div className="grid min-w-0 flex-1 grid-rows-[auto_auto_minmax(0,1fr)_auto] gap-4">
            <HeaderBar />

            {(errorMessage || noticeMessage || bootstrapping) && (
              <div className="grid gap-3">
                {bootstrapping && (
                  <Banner kind="info">
                    SmartHire is loading model metrics and preparing the workspace.
                  </Banner>
                )}
                {noticeMessage && <Banner kind="warning">{noticeMessage}</Banner>}
                {errorMessage && <Banner kind="error">{errorMessage}</Banner>}
              </div>
            )}

            <div className="grid min-h-[0] grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
              <main className="min-h-[0]">
                <SectionSwitch
                  activeSection={activeSection}
                  props={{
                    uploadedFile,
                    processing,
                    processed,
                    extractedText,
                    handleFileChange,
                    handleProcess,
                    language,
                    setLanguage,
                    chatMessages,
                    startInterview,
                    answerInput,
                    setAnswerInput,
                    submitAnswer,
                    feedback,
                    selectedModel,
                    setSelectedModel,
                    theme,
                    setTheme,
                    autoSave,
                    setAutoSave,
                    highlightEntities,
                    setHighlightEntities,
                    currentQuestion,
                    interviewStarting,
                    answerSubmitting,
                    parserLabel,
                    summary,
                    awaitingNextQuestion,
                    goToNextQuestion,
                  }}
                />
              </main>

              <aside className="flex min-h-[0] flex-col gap-4">
                <ProfileCard profile={profile} initials={initials} processed={processed} />
              </aside>
            </div>

            <BottomMetricsStrip metrics={metrics} />
          </div>
        </div>
      </div>
    </div>
  );
}

function Sidebar({ activeSection, onSelect }) {
  return (
    <aside
      className="flex w-[110px] shrink-0 flex-col rounded-[20px] px-3 py-4 shadow-card"
      style={{ backgroundColor: "#102a62", color: "#ffffff" }}
    >
      <div className="mb-4 flex h-11 items-center justify-center rounded-xl" style={{ color: "rgba(255,255,255,0.95)" }}>
        <Icon name="menu" className="h-5 w-5" />
      </div>

      <nav className="flex flex-1 flex-col gap-2">
        {NAV_ITEMS.map((item) => {
          const active = activeSection === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelect(item.id)}
              className="group flex min-h-[72px] flex-col items-center justify-center gap-2 rounded-2xl border px-2 py-3 text-center transition-all duration-200"
              style={{
                borderColor: active ? "rgba(255,255,255,0.15)" : "transparent",
                backgroundColor: active ? "rgba(255,255,255,0.10)" : "transparent",
                color: "#ffffff",
              }}
            >
              <span style={{ color: "#ffffff", opacity: active ? 1 : 0.88 }}>
                <Icon name={item.icon} className="h-[18px] w-[18px]" />
              </span>
              <span className="text-xs font-semibold tracking-wide" style={{ color: "rgba(255,255,255,0.96)" }}>
                {item.label}
              </span>
            </button>
          );
        })}
      </nav>
    </aside>
  );
}

function HeaderBar() {
  return (
    <header className="flex min-h-[112px] items-center rounded-[22px] border border-brand-100 bg-white px-8 py-6 shadow-card dark:border-slateish-700 dark:bg-slateish-900">
      <div>
        <h1 className="text-5xl font-extrabold tracking-tight text-slateish-700 dark:text-white">SmartHire</h1>
        <p className="mt-3 text-2xl font-medium text-slateish-400">Resume-to-Interview Assistant</p>
      </div>
    </header>
  );
}

function SectionSwitch({ activeSection, props }) {
  return (
    <div key={activeSection} className="section-fade h-full">
      {activeSection === "resume" && <ResumeView {...props} />}
      {activeSection === "interview" && <InterviewView {...props} />}
      {activeSection === "settings" && <SettingsView {...props} />}
    </div>
  );
}

function ResumeView({
  uploadedFile,
  processing,
  processed,
  extractedText,
  handleFileChange,
  handleProcess,
  parserLabel,
}) {
  return (
    <div className="grid h-full grid-cols-[minmax(0,360px)_minmax(0,1fr)] gap-4">
      <div className="flex flex-col gap-4">
        <SectionCard title="Uploaded Resume">
          <label className="flex cursor-pointer flex-col gap-4 rounded-[18px] border border-dashed px-5 py-5 transition"
            style={{
              borderColor: "rgba(16,42,98,0.12)",
              backgroundColor: "#f4f8ff",
            }}
          >
            <div className="flex items-center gap-3 text-slateish-600">
              <div
                className="flex h-14 w-14 items-center justify-center rounded-2xl border text-slateish-600 shadow-sm"
                style={{
                  borderColor: "rgba(16,42,98,0.12)",
                  backgroundColor: "#ffffff",
                }}
              >
                <Icon name="upload" className="h-6 w-6" />
              </div>
              <div className="min-w-0 flex-1">
                <div
                  className="break-all text-base font-bold leading-6"
                  style={{ color: "#102a62" }}
                >
                  {uploadedFile ? uploadedFile.name : "Upload Resume"}
                </div>
                <div className="mt-1 text-sm text-slateish-400">
                  {uploadedFile ? "Ready for extraction" : "PDF, DOCX, or TXT"}
                </div>
              </div>
            </div>
            <input type="file" className="hidden" accept=".pdf,.docx,.txt" onChange={handleFileChange} />
          </label>

          <button
            type="button"
            onClick={handleProcess}
            disabled={processing}
            className="mt-4 inline-flex min-h-[48px] items-center justify-center rounded-2xl px-5 text-base font-extrabold text-white shadow-card transition disabled:opacity-60"
            style={{ backgroundColor: "#102a62" }}
          >
            {processing ? <ProcessingInline label="Processing Resume" /> : "Process Document"}
          </button>
        </SectionCard>
      </div>

      <SectionCard title="Extracted Text" className="h-full">
        <div className="glass-scroll h-[640px] overflow-y-auto rounded-[20px] border border-brand-100 bg-brand-50 p-6 text-lg leading-9 text-slateish-500">
          {processing ? (
            <LoadingState label="Extracting text and structuring the resume..." />
          ) : (
            <pre className="whitespace-pre-wrap font-sans">{extractedText}</pre>
          )}
        </div>
      </SectionCard>
    </div>
  );
}

function InterviewView({
  processed,
  processing,
  language,
  setLanguage,
  chatMessages,
  startInterview,
  answerInput,
  setAnswerInput,
  submitAnswer,
  feedback,
  currentQuestion,
  interviewStarting,
  answerSubmitting,
  summary,
  awaitingNextQuestion,
  goToNextQuestion,
}) {
  return (
    <div className="grid h-full grid-cols-[minmax(0,1fr)] gap-4">
      <div className="rounded-[22px] border border-brand-100 bg-white p-6 shadow-card">
        {!processed ? (
          <EmptyStage
            title={language === "Arabic" ? "مساحة المقابلة" : "Interview Workspace"}
            copy={
              language === "Arabic"
                ? "ابدأ أولًا من صفحة السيرة، ثم عالج الملف، وبعدها ارجع هنا لبدء المقابلة."
                : "Process a resume first in the Resume section. After parsing finishes, come back here to start the interview."
            }
          />
        ) : (
          <div className="flex flex-col gap-4">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <h2 className="text-2xl font-extrabold text-slateish-700">
                  {language === "Arabic" ? "المقابلة الذكية" : "AI Interview"}
                </h2>
                <p className="mt-2 text-base text-slateish-400">
                  {language === "Arabic"
                    ? "مقابلة تفاعلية مع فيدباك بعد كل إجابة."
                    : "Chat-based interview flow with instant feedback after each answer."}
                </p>
              </div>

              <div className="flex items-center gap-3">
                <RadioPills label="Language" options={["English", "Arabic"]} value={language} onChange={setLanguage} />
                <button
                  type="button"
                  onClick={startInterview}
                  disabled={processing || interviewStarting}
                  className="rounded-2xl px-5 py-3 text-sm font-extrabold text-white transition disabled:cursor-not-allowed disabled:opacity-50"
                  style={{ backgroundColor: "#102a62" }}
                >
                  {interviewStarting ? (language === "Arabic" ? "جارٍ التحضير..." : "Preparing...") : (language === "Arabic" ? "ابدأ المقابلة" : "Start Interview")}
                </button>
              </div>
            </div>

            <div className="glass-scroll h-[470px] overflow-y-auto rounded-[20px] border border-brand-100 bg-slateish-50/40 p-5">
              {chatMessages.length === 0 ? (
                <EmptyStage
                  compact
                  title={language === "Arabic" ? "جاهز للبدء" : "Ready to Start"}
                  copy={
                    language === "Arabic"
                      ? "ابدأ المقابلة لترى الأسئلة، وإجاباتك، والفيدباك بعد كل إجابة."
                      : "Start the interview to see AI questions, your answers, and structured feedback."
                  }
                />
              ) : (
                <div className="space-y-4">
                  {chatMessages.map((message) => (
                    <ChatBubble key={message.id} message={message} language={language} />
                  ))}
                  {feedback && <FeedbackCard feedback={feedback} language={language} />}
                  {awaitingNextQuestion && (
                    <div className="flex justify-end">
                      <button
                        type="button"
                        onClick={goToNextQuestion}
                        className="rounded-2xl px-5 py-3 text-sm font-extrabold text-white transition"
                        style={{ backgroundColor: "#102a62" }}
                      >
                        {language === "Arabic" ? "السؤال التالي" : "Next Question"}
                      </button>
                    </div>
                  )}
                  {summary && (
                    <div className="rounded-[22px] border border-brand-100 bg-white p-5 shadow-card">
                      <div className="text-lg font-extrabold text-brand-500">
                        {language === "Arabic" ? "الملخص النهائي" : "Final Summary"}
                      </div>
                      <pre className="mt-3 whitespace-pre-wrap font-sans text-sm leading-8 text-slateish-500">{summary}</pre>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="rounded-[18px] border border-brand-100 bg-white p-4">
              <textarea
                value={answerInput}
                onChange={(event) => setAnswerInput(event.target.value)}
                placeholder={
                  currentQuestion
                    ? (language === "Arabic" ? "اكتب إجابتك هنا..." : "Type your answer here...")
                    : (language === "Arabic" ? "ابدأ المقابلة أولًا..." : "Start the interview first...")
                }
                className="h-28 w-full resize-none rounded-2xl border border-brand-100 bg-slateish-50 px-4 py-3 text-base text-slateish-600 outline-none transition focus:border-brand-300"
              />
              <div className="mt-3 flex justify-end">
                <button
                  type="button"
                  onClick={submitAnswer}
                  disabled={!currentQuestion || answerSubmitting}
                  className="rounded-2xl px-5 py-3 text-sm font-extrabold text-white transition disabled:opacity-50"
                  style={{ backgroundColor: "#102a62" }}
                >
                  {answerSubmitting ? (language === "Arabic" ? "جارٍ التقييم..." : "Evaluating...") : (language === "Arabic" ? "أرسل الإجابة" : "Send Answer")}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function SettingsView({
  language,
  setLanguage,
  selectedModel,
  setSelectedModel,
  theme,
  setTheme,
  autoSave,
  setAutoSave,
  highlightEntities,
  setHighlightEntities,
}) {
  return (
    <div className="grid gap-4">
      <SectionCard title="System Settings">
        <div className="grid gap-4 md:grid-cols-2">
          <PreferenceCard title="Language">
            <RadioPills label="Language" options={["English", "Arabic"]} value={language} onChange={setLanguage} />
          </PreferenceCard>

          <PreferenceCard title="Model Selection">
            <RadioPills
              label="Model"
              options={["BERT + Gemini", "BERT Only", "Gemini Only"]}
              value={selectedModel}
              onChange={setSelectedModel}
            />
            <p className="mt-3 text-sm leading-7 text-slateish-400">
              Resume extraction always uses the trained BERT parser. This option controls whether interview generation uses AI APIs or the offline fallback.
            </p>
          </PreferenceCard>

          <PreferenceCard title="Theme">
            <div className="flex gap-3">
              {["light", "dark"].map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => setTheme(mode)}
                  className={cx(
                    "rounded-2xl border px-5 py-3 text-sm font-bold transition",
                    theme === mode
                      ? "border-brand-300 bg-brand-50 text-brand-600"
                      : "border-brand-100 bg-white text-slateish-500 hover:bg-brand-50"
                  )}
                >
                  {mode === "light" ? "Light Theme" : "Dark Theme"}
                </button>
              ))}
            </div>
          </PreferenceCard>

          <PreferenceCard title="Preferences">
            <ToggleRow
              label="Auto-save interview history"
              checked={autoSave}
              onToggle={() => setAutoSave((value) => !value)}
            />
            <ToggleRow
              label="Highlight extracted entities"
              checked={highlightEntities}
              onToggle={() => setHighlightEntities((value) => !value)}
            />
          </PreferenceCard>
        </div>
      </SectionCard>
    </div>
  );
}

function SectionCard({ title, className = "", children }) {
  return (
    <section className={cx("rounded-[22px] border border-brand-100 bg-white p-6 shadow-card", className)}>
      <h3 className="mb-4 text-[1.1rem] font-extrabold text-slateish-700">{title}</h3>
      {children}
    </section>
  );
}

function Banner({ kind = "info", children }) {
  const styles = {
    info: "border-brand-200 bg-brand-50 text-brand-600",
    warning: "border-orange-200 bg-orange-50 text-orange-600",
    error: "border-rose-200 bg-rose-50 text-rose-600",
  };
  return <div className={cx("rounded-2xl border px-4 py-3 text-sm font-semibold", styles[kind])}>{children}</div>;
}

function ProfileCard({ profile, initials, processed }) {
  return (
    <section className="rounded-[22px] border border-brand-100 bg-white p-6 shadow-card">
      <h3 className="text-[1.1rem] font-extrabold text-slateish-700">Resume Intelligence</h3>
      <p className="mt-1 text-sm font-medium text-slateish-400">Structured with BERT NER</p>

      <div className="mt-6 flex flex-col items-start gap-4 text-left">
        <div className="flex h-20 w-20 shrink-0 items-center justify-center rounded-full bg-[#9ec5ff] text-white shadow-card">
          <Icon name="profile" className="h-10 w-10" />
        </div>
        <div className="min-w-0 w-full text-left">
          <div className="text-2xl font-extrabold tracking-tight text-slateish-700 break-words text-left">
            {profile.name}
          </div>
          <div className="mt-2 text-lg font-medium text-slateish-400 break-words text-left">
            {profile.title}
          </div>
        </div>
      </div>

      <div className="mt-6 flex items-center justify-between">
        <span className="text-lg font-bold text-slateish-400">Role Match</span>
        <span className="rounded-full bg-emerald-100 px-4 py-2 text-lg font-extrabold text-emerald-600">
          {processed ? `${profile.roleMatch}%` : "--"}
        </span>
      </div>

      <div className="mt-6 grid grid-cols-[110px_minmax(0,1fr)] gap-y-4 text-[1.02rem]">
        <span className="font-bold text-slateish-400">Location</span>
        <span className="font-medium text-slateish-600">{profile.location}</span>
        <span className="font-bold text-slateish-400">Email</span>
        <span className="font-medium break-all text-slateish-600">{profile.email}</span>
        <span className="font-bold text-slateish-400">Phone</span>
        <span className="font-medium text-slateish-600">{profile.phone}</span>
      </div>

      <div className="mt-6 border-t border-brand-100 pt-5">
        <div className="text-base font-extrabold text-slateish-700">Resume Intelligence Summary</div>
        <p className="mt-3 text-sm leading-7 text-slateish-500">{profile.summary}</p>
      </div>

      {profile.about && profile.about !== "Not detected" ? (
        <div className="mt-6 border-t border-brand-100 pt-5">
          <div className="text-base font-extrabold text-slateish-700">Professional Summary</div>
          <p className="mt-3 text-sm leading-7 text-slateish-500">{profile.about}</p>
        </div>
      ) : null}

      <div className="mt-6 border-t border-brand-100 pt-5">
        <div className="text-base font-extrabold text-slateish-700">Education</div>
        <div className="mt-3 grid grid-cols-[110px_minmax(0,1fr)] gap-y-3 text-[0.98rem]">
          <span className="font-bold text-slateish-400">University</span>
          <span className="font-medium text-slateish-600">{profile.university}</span>
          <span className="font-bold text-slateish-400">Major</span>
          <span className="font-medium text-slateish-600">{profile.major || profile.education}</span>
          <span className="font-bold text-slateish-400">Qualification</span>
          <span className="font-medium text-slateish-600">{profile.education}</span>
          <span className="font-bold text-slateish-400">Graduation</span>
          <span className="font-medium text-slateish-600">{profile.graduation}</span>
        </div>
      </div>

      <div className="mt-6 border-t border-brand-100 pt-5">
        <div className="text-base font-extrabold text-slateish-700">Experience</div>
        <div className="mt-3 grid grid-cols-[110px_minmax(0,1fr)] gap-y-3 text-[0.98rem]">
          <span className="font-bold text-slateish-400">Current Role</span>
          <span className="font-medium text-slateish-600">{profile.title}</span>
          <span className="font-bold text-slateish-400">Company</span>
          <span className="font-medium text-slateish-600">{profile.company}</span>
          <span className="font-bold text-slateish-400">Duration</span>
          <span className="font-medium text-slateish-600">{profile.experienceLength}</span>
        </div>
        {profile.experienceHighlights?.length ? (
          <div className="mt-4 rounded-2xl border border-brand-100 bg-slateish-50 px-4 py-4">
            <div className="text-sm font-extrabold uppercase tracking-[0.18em] text-slateish-400">Experience Highlights</div>
            <ul className="mt-3 space-y-2 text-sm leading-6 text-slateish-600">
              {profile.experienceHighlights.map((item) => (
                <li key={item} className="flex gap-2">
                  <span className="mt-[9px] h-1.5 w-1.5 shrink-0 rounded-full bg-brand-500"></span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>

      {(profile.languages?.length || profile.projects?.length) ? (
        <div className="mt-6 border-t border-brand-100 pt-5">
          {profile.languages?.length ? (
            <div className="mb-4">
              <div className="text-base font-extrabold text-slateish-700">Languages</div>
              <div className="mt-3 flex flex-wrap gap-2">
                {profile.languages.map((language) => (
                  <span key={language} className="rounded-full border border-emerald-100 bg-emerald-50 px-3 py-2 text-sm font-bold text-emerald-600">
                    {language}
                  </span>
                ))}
              </div>
            </div>
          ) : null}

          {profile.projects?.length ? (
            <div>
              <div className="text-base font-extrabold text-slateish-700">Projects</div>
              <div className="mt-3 space-y-2">
                {profile.projects.map((project) => (
                  <div key={project} className="rounded-2xl border border-brand-100 bg-slateish-50 px-4 py-3 text-sm font-medium text-slateish-600">
                    {project}
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      ) : null}

      <div className="mt-6 border-t border-brand-100 pt-5">
        <div className="mb-3 text-base font-extrabold text-slateish-700">
          All Extracted Skills
          {profile.skills.length ? <span className="ml-2 text-sm font-semibold text-slateish-400">({profile.skills.length})</span> : null}
        </div>
        <div className="pr-1">
          <div className="flex flex-wrap gap-2">
            {profile.skills.length ? (
              profile.skills.map((skill) => (
                <span key={skill} className="rounded-full border border-brand-100 bg-brand-50 px-3 py-2 text-sm font-bold text-brand-600">
                  {skill}
                </span>
              ))
            ) : (
              <span className="text-sm text-slateish-400">Skills will appear after parsing the resume.</span>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

function BottomMetricsStrip({ metrics }) {
  return (
    <section className="rounded-[22px] border border-brand-100 bg-white px-4 py-4 shadow-card">
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {metrics.map((item) => (
          <div
            key={item.title}
            className="min-w-0 rounded-[18px] border border-brand-100 bg-white px-5 py-4 transition"
          >
            <div className="text-[0.72rem] font-extrabold uppercase tracking-[0.22em] text-slateish-400">
              {item.title}
            </div>
            <div
              className={cx(
                "mt-3 break-words font-extrabold leading-tight",
                item.smaller ? "text-[1.9rem]" : "text-[2.45rem]",
                item.color
              )}
            >
              {item.value}
            </div>
            <div className="mt-2 text-sm font-semibold text-slateish-500">
              {item.note}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function ChatBubble({ message, language = "English" }) {
  const isUser = message.role === "user";
  const bubbleText =
    message.type === "question"
      ? null
      : message.type === "summary"
        ? message.content
        : message.content;

  return (
    <div className={cx("chat-enter flex items-start gap-3", isUser && "justify-end")}>
      {!isUser && (
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-[#9ec5ff] text-white shadow-sm">
          <Icon name="robot" className="h-5 w-5" />
        </div>
      )}

      <div
        className={cx(
          "max-w-[78%] rounded-[22px] border border-brand-100 px-5 py-4 shadow-card",
          isUser ? "bg-[#eef4ff] text-slateish-600" : "bg-white text-slateish-600"
        )}
      >
        {message.type === "question" ? (
          <div>
            <div className="text-sm font-extrabold text-brand-500">
              {language === "Arabic" ? "السؤال" : "Question"}
            </div>
            <div className="mt-2 text-lg font-bold text-slateish-700">
              {message.question.type}: {message.question.question}
            </div>
            <div className="mt-3 text-sm font-medium text-slateish-400">
              {language === "Arabic" ? "بناءً على:" : "Based on:"} {message.question.basedOn}
            </div>
          </div>
        ) : message.type === "summary" ? (
          <div>
            <div className="text-sm font-extrabold text-emerald-500">
              {language === "Arabic" ? "الملخص النهائي" : "Final Summary"}
            </div>
            <pre className="mt-3 whitespace-pre-wrap font-sans text-sm leading-8 text-slateish-500">{bubbleText}</pre>
          </div>
        ) : (
          <div className="text-lg leading-8">{bubbleText}</div>
        )}
        <div className="mt-3 text-xs font-medium text-slateish-300">{message.time}</div>
      </div>

      {isUser && (
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-[#9ec5ff] text-white shadow-sm">
          <Icon name="profile" className="h-5 w-5" />
        </div>
      )}
    </div>
  );
}

function FeedbackCard({ feedback, language = "English" }) {
  const circumference = 2 * Math.PI * 42;
  const offset = circumference - (feedback.score / 5) * circumference;

  return (
    <div className="chat-enter rounded-[22px] border border-brand-100 bg-white p-6 shadow-card">
      <div className="mb-4 text-2xl font-extrabold text-emerald-500">
        {language === "Arabic" ? "التغذية الراجعة" : "Feedback"}
      </div>
      <div className="grid gap-6 lg:grid-cols-[180px_repeat(3,minmax(0,1fr))]">
        <div>
          <div className="text-sm font-bold text-slateish-400">{language === "Arabic" ? "الدرجة" : "Score"}</div>
          <div className="mt-4 flex items-center justify-center">
            <div className="relative h-32 w-32">
              <svg viewBox="0 0 100 100" className="h-32 w-32 -rotate-90">
                <circle cx="50" cy="50" r="42" stroke="#dbf5e8" strokeWidth="8" fill="none" />
                <circle
                  cx="50"
                  cy="50"
                  r="42"
                  stroke="#22c16b"
                  strokeWidth="8"
                  fill="none"
                  strokeLinecap="round"
                  strokeDasharray={circumference}
                  strokeDashoffset={offset}
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center text-4xl font-extrabold text-emerald-500">
                {feedback.score}/5
              </div>
            </div>
          </div>
        </div>

        <FeedbackColumn title={language === "Arabic" ? "نقاط القوة" : "Strengths"} color="text-emerald-500" copy={feedback.strengths} />
        <FeedbackColumn title={language === "Arabic" ? "التحسينات" : "Improvements"} color="text-orange-500" copy={feedback.improvements} />
        <FeedbackColumn title={language === "Arabic" ? "إجابة مقترحة" : "Suggested Answer"} color="text-brand-500" copy={feedback.suggested} />
      </div>
    </div>
  );
}

function FeedbackColumn({ title, color, copy }) {
  return (
    <div>
      <div className={cx("text-sm font-extrabold", color)}>{title}</div>
      <p className="mt-3 text-sm leading-7 text-slateish-400">{copy}</p>
    </div>
  );
}

function PreferenceCard({ title, children }) {
  return (
    <div className="rounded-[22px] border border-brand-100 bg-brand-50 p-5">
      <div className="mb-4 text-lg font-extrabold text-slateish-700">{title}</div>
      {children}
    </div>
  );
}

function ToggleRow({ label, checked, onToggle }) {
  return (
    <button
      type="button"
      onClick={onToggle}
      className="mb-3 flex w-full items-center justify-between rounded-2xl border border-brand-100 bg-white px-4 py-4 text-left transition hover:bg-brand-50"
    >
      <span className="font-semibold text-slateish-600">{label}</span>
      <span
        className={cx(
          "flex h-7 w-12 items-center rounded-full px-1 transition",
          checked ? "justify-end bg-brand-500" : "justify-start bg-slateish-200"
        )}
      >
        <span className="h-5 w-5 rounded-full bg-white shadow-sm" />
      </span>
    </button>
  );
}

function RadioPills({ label, options, value, onChange }) {
  return (
    <div className="flex flex-wrap gap-3">
      {options.map((option) => {
        const active = option === value;
        return (
          <button
            key={option}
            type="button"
            onClick={() => onChange(option)}
            aria-label={`${label}: ${option}`}
            className="rounded-full border px-4 py-3 text-sm font-extrabold transition"
            style={{
              borderColor: active ? "#102a62" : "rgba(16,42,98,0.14)",
              backgroundColor: active ? "#102a62" : "#ffffff",
              color: active ? "#ffffff" : "#102a62",
            }}
          >
            {option}
          </button>
        );
      })}
    </div>
  );
}

function ProcessingInline({ label }) {
  return (
    <span className="inline-flex items-center gap-3">
      <span className="flex items-center gap-1">
        <span className="pulse-ring h-2.5 w-2.5 rounded-full bg-white/90" />
        <span className="pulse-ring h-2.5 w-2.5 rounded-full bg-white/80" style={{ animationDelay: "0.18s" }} />
        <span className="pulse-ring h-2.5 w-2.5 rounded-full bg-white/70" style={{ animationDelay: "0.36s" }} />
      </span>
      <span>{label}</span>
    </span>
  );
}

function LoadingState({ label }) {
  return (
    <div className="flex h-full min-h-[220px] flex-col items-center justify-center gap-5 text-center">
      <div className="pulse-ring flex h-16 w-16 items-center justify-center rounded-full bg-white text-brand-500 shadow-card">
        <Icon name="spark" className="h-8 w-8" />
      </div>
      <div>
        <div className="text-lg font-extrabold text-slateish-700">SmartHire is Processing</div>
        <div className="mt-2 text-sm text-slateish-400">{label}</div>
      </div>
    </div>
  );
}

function EmptyStage({ title, copy, compact = false }) {
  return (
    <div className={cx("flex flex-col items-center justify-center text-center", compact ? "min-h-[260px]" : "min-h-[420px]")}>
      <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-brand-50 text-brand-500">
        <Icon name="spark" className="h-8 w-8" />
      </div>
      <h3 className="text-2xl font-extrabold text-slateish-700">{title}</h3>
      <p className="mt-4 max-w-xl text-lg leading-8 text-slateish-400">{copy}</p>
    </div>
  );
}

function Icon({ name, className = "h-5 w-5" }) {
  const common = {
    className,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: "2",
    strokeLinecap: "round",
    strokeLinejoin: "round",
  };

  if (name === "menu") {
    return (
      <svg {...common}>
        <path d="M4 7h16M4 12h16M4 17h16" />
      </svg>
    );
  }
  if (name === "resume") {
    return (
      <svg {...common}>
        <path d="M8 3h7l4 4v14H8a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Z" />
        <path d="M15 3v5h5M10 12h4M10 16h4" />
      </svg>
    );
  }
  if (name === "interview") {
    return (
      <svg {...common}>
        <path d="M7 8h10M7 12h6M8 18l-3 2 .5-3A6 6 0 0 1 7 4h10a5 5 0 0 1 5 5v2a5 5 0 0 1-5 5H9" />
      </svg>
    );
  }
  if (name === "settings") {
    return (
      <svg {...common}>
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.7 1.7 0 0 0 .34 1.87l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.7 1.7 0 0 0-1.87-.34 1.7 1.7 0 0 0-1.04 1.56V22a2 2 0 1 1-4 0v-.09a1.7 1.7 0 0 0-1.12-1.58 1.7 1.7 0 0 0-1.79.36l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.7 1.7 0 0 0 .34-1.87 1.7 1.7 0 0 0-1.56-1.04H2a2 2 0 1 1 0-4h.09A1.7 1.7 0 0 0 3.65 9.8a1.7 1.7 0 0 0-.36-1.79l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.7 1.7 0 0 0 1.87.34H8a1.7 1.7 0 0 0 1-1.52V2a2 2 0 1 1 4 0v.09a1.7 1.7 0 0 0 1.04 1.56 1.7 1.7 0 0 0 1.87-.34l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.7 1.7 0 0 0-.34 1.87v.08A1.7 1.7 0 0 0 21.91 11H22a2 2 0 1 1 0 4h-.09a1.7 1.7 0 0 0-1.56 1.04Z" />
      </svg>
    );
  }
  if (name === "robot") {
    return (
      <svg {...common}>
        <rect x="7" y="8" width="10" height="8" rx="2" />
        <path d="M12 4v3M9 16v2M15 16v2M5 11H3M21 11h-2" />
        <circle cx="10" cy="12" r="1" fill="currentColor" stroke="none" />
        <circle cx="14" cy="12" r="1" fill="currentColor" stroke="none" />
      </svg>
    );
  }
  if (name === "profile") {
    return (
      <svg {...common}>
        <circle cx="12" cy="8" r="3.2" />
        <path d="M6.5 19a5.8 5.8 0 0 1 11 0" />
      </svg>
    );
  }
  if (name === "upload") {
    return (
      <svg {...common}>
        <path d="M12 16V4M7 9l5-5 5 5M5 20h14" />
      </svg>
    );
  }
  if (name === "spark") {
    return (
      <svg {...common}>
        <path d="m12 3 1.8 4.2L18 9l-4.2 1.8L12 15l-1.8-4.2L6 9l4.2-1.8L12 3Z" />
      </svg>
    );
  }
  return null;
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
