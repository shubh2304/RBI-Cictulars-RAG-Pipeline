"use client";

import React, { useState, useEffect, useRef } from "react";
import { 
  Send, 
  FileText, 
  ExternalLink, 
  Search, 
  Copy, 
  Check, 
  Info, 
  AlertTriangle, 
  Menu, 
  X, 
  Library, 
  HelpCircle, 
  ChevronRight, 
  Loader2 
} from "lucide-react";

interface DocumentMeta {
  filename: string;
  document_name: string;
  document_type: string;
  ref_number: string | null;
  circular_number: string | null;
  pub_date: string | null;
  source_url: string | null;
}

interface CitationSource {
  document_name: string;
  filename: string;
  source_pdf_path: string;
  page_number: number;
  section_title: string;
  circular_number: string | null;
  ref_number: string | null;
  matched_sentence: string;
}

interface Citation {
  citation_tag: string;
  statement: string;
  verified: boolean;
  pdf_url: string;
  source: CitationSource;
  scores: {
    semantic_similarity: number;
    jaccard_overlap: number;
  };
}

interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  warnings?: string[];
  hallucination_detected?: boolean;
  answerable?: boolean;
}

const BACKEND_URL = "http://localhost:8000";

export default function RbiRagDashboard() {
  // Sidebar state
  const [documents, setDocuments] = useState<DocumentMeta[]>([]);
  const [searchDocQuery, setSearchDocQuery] = useState("");
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  
  // Chat state
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  
  // PDF Viewer state
  const [activePdf, setActivePdf] = useState<string | null>(null);
  const [activePage, setActivePage] = useState<number>(1);
  const [viewerKey, setViewerKey] = useState(0); // Used to force iframe re-mount
  
  // Copy to clipboard helper states
  const [copiedText, setCopiedText] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Fetch document list on load
  useEffect(() => {
    fetchDocuments();
  }, []);

  // Auto-scroll chat to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/documents`);
      if (!res.ok) throw new Error("Failed to load documents list.");
      const data = await res.json();
      setDocuments(data);
    } catch (err: any) {
      console.error("Error loading documents:", err);
    }
  };

  const handleQuerySubmit = async (e?: React.FormEvent, customQuery?: string) => {
    if (e) e.preventDefault();
    const queryToSend = customQuery || query;
    if (!queryToSend.trim() || loading) return;

    setLoading(true);
    setErrorMessage(null);
    setQuery("");

    // Add user message
    const userMsg: Message = { role: "user", content: queryToSend };
    setMessages((prev) => [...prev, userMsg]);

    try {
      const response = await fetch(`${BACKEND_URL}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: queryToSend }),
      });

      if (!response.ok) {
        throw new Error(`API responded with status code ${response.status}`);
      }

      const data = await response.json();
      
      const assistantMsg: Message = {
        role: "assistant",
        content: data.response,
        citations: data.citations || [],
        warnings: data.warnings || [],
        hallucination_detected: data.hallucination_detected || false,
        answerable: data.answerable ?? true
      };

      setMessages((prev) => [...prev, assistantMsg]);
      
      // Auto-load first verified citation PDF if available
      if (data.citations && data.citations.length > 0) {
        const firstCit = data.citations[0];
        if (firstCit.source?.filename) {
          selectPdf(firstCit.source.filename, firstCit.source.page_number);
        }
      }
    } catch (err: any) {
      console.error("Query execution error:", err);
      setErrorMessage(
        "Could not connect to the local RAG API server. Please verify uvicorn api_server is running on http://localhost:8000"
      );
    } finally {
      setLoading(false);
    }
  };

  const selectPdf = (filename: string, pageNum: number) => {
    setActivePdf(filename);
    setActivePage(pageNum);
    setViewerKey((prev) => prev + 1); // increment key to force React iframe unmount-remount
  };

  const handleCopyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedText(id);
    setTimeout(() => setCopiedText(null), 2000);
  };

  // Render inline rich references [1: Master Circular, Page 5] as clickable links
  const renderResponseWithLinks = (text: string, citations: Citation[]) => {
    if (!text) return "";
    
    // Regular expression matching: [N: Document Name, Page X, Section Y] or [N: Document Name, Page X]
    const regex = /\[(\d+)\:\s*([^,\]]+)\s*,\s*Page\s*(\d+)(?:,\s*Section\s*([^\]]+))?\]/g;
    
    const parts = [];
    let lastIndex = 0;
    let match;
    
    while ((match = regex.exec(text)) !== null) {
      const matchIndex = match.index;
      if (matchIndex > lastIndex) {
        parts.push(text.substring(lastIndex, matchIndex));
      }
      
      const tagNum = parseInt(match[1]);
      const docName = match[2];
      const pageNum = parseInt(match[3]);
      
      // Attempt to resolve file name from corresponding citation tag
      const matchingCit = citations.find((c) => {
        const tagDigits = c.citation_tag.replace(/[^\d]/g, "");
        return parseInt(tagDigits) === tagNum;
      });
      
      const filename = matchingCit?.source?.filename || "";
      
      parts.push(
        <button
          key={matchIndex}
          onClick={() => {
            if (filename) {
              selectPdf(filename, pageNum);
            }
          }}
          className="inline-flex items-center px-1.5 py-0.5 mx-0.5 rounded text-xs font-semibold bg-indigo-950/60 text-indigo-300 border border-indigo-800/80 hover:bg-indigo-900 hover:text-indigo-200 transition-colors cursor-pointer"
          title={`Click to open ${docName} at Page ${pageNum}`}
        >
          [{tagNum}: Page {pageNum}]
        </button>
      );
      
      lastIndex = regex.lastIndex;
    }
    
    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }
    
    return parts.length > 0 ? parts : text;
  };

  // Filter list of circulars based on search bar text
  const filteredDocuments = documents.filter((doc) =>
    doc.document_name.toLowerCase().includes(searchDocQuery.toLowerCase()) ||
    doc.filename.toLowerCase().includes(searchDocQuery.toLowerCase())
  );

  const suggestedPrompts = [
    "What is the agriculture target under PSL?",
    "What is the limit for collateral-free agricultural loans?",
    "Are retail and wholesale trade included in the MSME sector for priority sector lending?",
    "What is the Block Level Bankers Committee constitution and frequency under Lead Bank Scheme?"
  ];

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#050811] text-slate-100 antialiased">
      {/* 1. DOCUMENT LIBRARY SIDEBAR */}
      <div 
        className={`glass-panel border-r transition-all duration-300 flex flex-col h-full z-20 ${
          isSidebarOpen ? "w-80" : "w-0 overflow-hidden border-r-0"
        }`}
      >
        <div className="p-4 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center space-x-2 text-indigo-400 font-bold text-sm tracking-wider uppercase">
            <Library size={18} />
            <span>Documents Library</span>
          </div>
          <button 
            onClick={() => setIsSidebarOpen(false)}
            className="text-slate-400 hover:text-slate-100 p-1 hover:bg-slate-800 rounded transition-colors cursor-pointer"
          >
            <X size={18} />
          </button>
        </div>

        {/* Search documents */}
        <div className="p-3 border-b border-slate-800/60">
          <div className="relative flex items-center">
            <Search className="absolute left-3 text-slate-500" size={16} />
            <input
              type="text"
              placeholder="Search circulars..."
              value={searchDocQuery}
              onChange={(e) => setSearchDocQuery(e.target.value)}
              className="w-full bg-slate-950/80 border border-slate-800/80 rounded-md py-1.5 pl-9 pr-3 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500/80 transition-colors"
            />
          </div>
        </div>

        {/* Circulars List */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {filteredDocuments.length === 0 ? (
            <div className="text-center py-8 text-xs text-slate-500">
              No matching documents found.
            </div>
          ) : (
            filteredDocuments.map((doc, i) => (
              <button
                key={i}
                onClick={() => selectPdf(doc.filename, 1)}
                className={`w-full text-left p-2.5 rounded-md text-xs hover:bg-slate-800/50 hover:border-slate-700 border transition-all cursor-pointer block ${
                  activePdf === doc.filename 
                    ? "bg-indigo-950/40 border-indigo-800 text-slate-100" 
                    : "bg-transparent border-transparent text-slate-400"
                }`}
              >
                <div className="font-semibold line-clamp-2 leading-relaxed mb-1 text-slate-200">
                  {doc.document_name}
                </div>
                <div className="flex items-center space-x-2 text-[10px] text-slate-500 font-mono">
                  <span className="bg-slate-900 px-1 py-0.5 rounded text-indigo-400 border border-slate-800">
                    {doc.document_type}
                  </span>
                  <span>{doc.pub_date || "Circular"}</span>
                </div>
              </button>
            ))
          )}
        </div>
        
        {/* Statistics info */}
        <div className="p-3 border-t border-slate-800 bg-slate-950/40 text-[10px] text-slate-500 font-mono text-center">
          Total Seeded Circulars: {documents.length}
        </div>
      </div>

      {/* Toggle Sidebar Icon (when closed) */}
      {!isSidebarOpen && (
        <button
          onClick={() => setIsSidebarOpen(true)}
          className="absolute top-4 left-4 z-30 p-2 bg-slate-900 border border-slate-800 text-slate-300 hover:text-slate-100 hover:bg-slate-800 rounded-md transition-all cursor-pointer shadow-md"
          title="Open Documents Sidebar"
        >
          <Library size={18} />
        </button>
      )}

      {/* MAIN CONTENT SPLIT LAYOUT */}
      <div className="flex-1 flex overflow-hidden relative">
        
        {/* 2. CHAT & CITATIONS PANEL (LEFT SPLIT) */}
        <div className="w-1/2 flex flex-col h-full border-r border-slate-800/60 bg-[#070a14]">
          {/* Header */}
          <div className="h-14 border-b border-slate-850 px-6 flex items-center justify-between bg-slate-950/50 select-none">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 rounded bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center font-bold text-white shadow-lg text-sm">
                RBI
              </div>
              <div>
                <h1 className="text-sm font-semibold text-slate-200 leading-tight">RBI Compliance RAG Pipeline</h1>
                <div className="flex items-center space-x-1.5">
                  <span className="inline-block w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span>
                  <span className="text-[10px] text-slate-400 font-medium">Local Qwen2.5 7B Instruct</span>
                </div>
              </div>
            </div>
            
            {/* Show error notification if API server is offline */}
            {errorMessage && (
              <div className="flex items-center space-x-1.5 text-xs text-rose-400 bg-rose-950/30 px-2.5 py-1 rounded border border-rose-900/60 max-w-sm truncate animate-bounce">
                <AlertTriangle size={14} />
                <span>Backend offline</span>
              </div>
            )}
          </div>

          {/* Chat scrollable container */}
          <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
            {messages.length === 0 ? (
              <div className="h-full flex flex-col justify-center items-center py-12 max-w-lg mx-auto text-center select-none">
                <div className="w-12 h-12 rounded-full bg-indigo-950/40 text-indigo-400 border border-indigo-900/40 flex items-center justify-center mb-4">
                  <HelpCircle size={24} />
                </div>
                <h2 className="text-lg font-bold text-slate-200 mb-2">How can I assist you today?</h2>
                <p className="text-xs text-slate-400 leading-relaxed mb-6">
                  Query Reserve Bank of India (RBI) circulars, notifications, and directions. Citations and links point dynamically to the exact source pages.
                </p>
                
                {/* Suggested prompts list */}
                <div className="grid grid-cols-1 gap-2 w-full">
                  {suggestedPrompts.map((p, i) => (
                    <button
                      key={i}
                      onClick={() => handleQuerySubmit(undefined, p)}
                      className="text-left text-xs p-3 rounded-md bg-slate-900/60 border border-slate-800/80 text-slate-300 hover:bg-slate-800/40 hover:text-slate-100 hover:border-slate-700 hover:scale-[1.01] active:scale-[0.99] transition-all cursor-pointer flex items-center justify-between group"
                    >
                      <span className="line-clamp-1">{p}</span>
                      <ChevronRight size={14} className="text-slate-500 group-hover:text-indigo-400 group-hover:translate-x-0.5 transition-all flex-shrink-0 ml-2" />
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((msg, i) => (
                <div 
                  key={i} 
                  className={`flex flex-col space-y-2 animate-fade-in ${
                    msg.role === "user" ? "items-end" : "items-start"
                  }`}
                >
                  {/* Sender Label */}
                  <div className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide px-1">
                    {msg.role === "user" ? "Compliance Query" : "System Response"}
                  </div>

                  {/* Message bubble */}
                  <div 
                    className={`max-w-[90%] rounded-lg p-4 text-sm leading-relaxed ${
                      msg.role === "user" 
                        ? "bg-indigo-600 text-white shadow-md rounded-tr-none" 
                        : "glass-panel bg-slate-900/60 border-slate-800 text-slate-200 rounded-tl-none"
                    }`}
                  >
                    {msg.role === "user" ? (
                      msg.content
                    ) : (
                      renderResponseWithLinks(msg.content, msg.citations || [])
                    )}
                  </div>

                  {/* Warnings or Hallucination detection */}
                  {msg.role === "assistant" && msg.hallucination_detected && (
                    <div className="flex items-start space-x-2 text-xs text-amber-400 bg-amber-950/20 px-3 py-2 rounded-md border border-amber-900/50 max-w-[90%]">
                      <AlertTriangle size={16} className="mt-0.5 flex-shrink-0" />
                      <div>
                        <div className="font-semibold">Hallucination Pre-warning:</div>
                        <ul className="list-disc pl-4 space-y-0.5 mt-0.5 text-[11px] text-slate-400">
                          {msg.warnings?.map((w, idx) => (
                            <li key={idx}>{w}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  )}

                  {/* Clean citations breakdown panel */}
                  {msg.role === "assistant" && msg.citations && msg.citations.length > 0 && (
                    <div className="w-full max-w-[90%] space-y-2 mt-1">
                      <div className="text-[10px] font-semibold text-slate-500 tracking-wider uppercase px-1">
                        Source Citations
                      </div>
                      
                      <div className="space-y-1.5">
                        {msg.citations.map((cit, idx) => {
                          const tagDigits = cit.citation_tag.replace(/[^\d]/g, "");
                          const src = cit.source;
                          const verified = cit.verified;
                          
                          return (
                            <div 
                              key={idx}
                              className={`p-2.5 rounded-md text-xs border transition-all ${
                                verified 
                                  ? "bg-slate-900/30 border-slate-800/80 hover:border-slate-700" 
                                  : "bg-rose-950/10 border-rose-900/30 hover:border-rose-900/50"
                              }`}
                            >
                              <div className="flex items-center justify-between mb-1.5">
                                <div className="flex items-center space-x-2">
                                  <span className="w-5 h-5 rounded flex items-center justify-center font-bold text-[10px] bg-indigo-950 text-indigo-400 border border-indigo-800">
                                    {tagDigits}
                                  </span>
                                  <span className="font-bold text-slate-300">
                                    {src.document_name}
                                  </span>
                                </div>
                                <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded ${
                                  verified 
                                    ? "bg-green-950/60 text-green-400 border border-green-900/60" 
                                    : "bg-rose-950/60 text-rose-400 border border-rose-900/60 animate-pulse"
                                }`}>
                                  {verified ? "VERIFIED" : "WARNING"}
                                </span>
                              </div>

                              {/* Exact sentence snippet matched */}
                              <div className="text-slate-400 leading-relaxed bg-slate-950/60 p-2 rounded border border-slate-850/60 font-serif italic relative group pr-8">
                                "{src.matched_sentence || cit.statement}"
                                <button
                                  onClick={() => handleCopyToClipboard(src.matched_sentence || cit.statement, `cit-text-${idx}`)}
                                  className="absolute top-2 right-2 text-slate-500 hover:text-slate-200 p-0.5 hover:bg-slate-800 rounded transition-colors"
                                  title="Copy text snippet to clipboard (Use Ctrl+F to search in PDF)"
                                >
                                  {copiedText === `cit-text-${idx}` ? (
                                    <Check size={12} className="text-green-400" />
                                  ) : (
                                    <Copy size={12} />
                                  )}
                                </button>
                              </div>

                              {/* Details and Page Link clicker */}
                              <div className="flex items-center justify-between mt-2 text-[10px] text-slate-500 font-mono">
                                <div>
                                  Page: <span className="text-slate-300 font-semibold">{src.page_number}</span>
                                  {src.section_title && src.section_title !== "None" && (
                                    <> | Sec: <span className="text-slate-300 font-semibold">{src.section_title}</span></>
                                  )}
                                </div>
                                <button
                                  onClick={() => selectPdf(src.filename, src.page_number)}
                                  className="flex items-center space-x-1 text-indigo-400 hover:text-indigo-300 transition-colors font-sans font-semibold cursor-pointer"
                                >
                                  <span>View source page</span>
                                  <ChevronRight size={10} />
                                </button>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              ))
            )}
            
            {/* Loading shimmer bubble */}
            {loading && (
              <div className="flex flex-col space-y-2 items-start animate-pulse">
                <div className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide px-1">
                  RAG Pipeline Working...
                </div>
                <div className="glass-panel w-full max-w-[85%] rounded-lg p-4 space-y-2.5 rounded-tl-none border-slate-800">
                  <div className="h-4 shimmer rounded w-1/3"></div>
                  <div className="h-3.5 shimmer rounded w-full"></div>
                  <div className="h-3.5 shimmer rounded w-[92%]"></div>
                  <div className="h-3.5 shimmer rounded w-[75%]"></div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Chat Form */}
          <div className="p-4 border-t border-slate-850/80 bg-slate-950/20">
            <form onSubmit={handleQuerySubmit} className="relative flex items-center">
              <input
                type="text"
                placeholder="Ask about RBI circulars, targets, limits, or regulations..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                disabled={loading}
                className="w-full bg-slate-950/80 border border-slate-800 rounded-lg py-3.5 pl-4 pr-12 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/20 transition-all disabled:opacity-50"
              />
              <button
                type="submit"
                disabled={loading || !query.trim()}
                className="absolute right-3 p-2 rounded-md bg-indigo-600 hover:bg-indigo-500 text-white transition-all disabled:opacity-40 disabled:hover:bg-indigo-600 cursor-pointer flex items-center justify-center shadow-md active:scale-95"
              >
                {loading ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
              </button>
            </form>
          </div>
        </div>

        {/* 3. DYNAMIC PDF VIEWER PANEL (RIGHT SPLIT) */}
        <div className="w-1/2 flex flex-col h-full bg-[#05070e] relative">
          
          {/* Header metadata bar */}
          <div className="h-14 border-b border-slate-850 px-6 flex items-center justify-between bg-slate-950/40 select-none">
            <div className="flex items-center space-x-2 text-slate-300 font-semibold text-xs truncate">
              <FileText size={16} className="text-indigo-400 flex-shrink-0" />
              <span className="truncate">{activePdf ? activePdf : "PDF Document Viewer"}</span>
            </div>
            
            {activePdf && (
              <div className="flex items-center space-x-3">
                <span className="bg-slate-900 border border-slate-800 px-2 py-0.5 rounded text-[10px] font-mono text-slate-400">
                  Target Page: {activePage}
                </span>
                
                {/* External link to open PDF in standard tab */}
                <a
                  href={`${BACKEND_URL}/api/document/${encodeURIComponent(activePdf)}#page=${activePage}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center space-x-1 text-[11px] text-indigo-400 hover:text-indigo-300 font-semibold transition-colors"
                  title="Open source PDF in native browser tab (recommended for page anchors)"
                >
                  <span>Open tab</span>
                  <ExternalLink size={12} />
                </a>
              </div>
            )}
          </div>

          {/* Iframe View */}
          <div className="flex-1 p-4 bg-slate-950/80">
            {activePdf ? (
              <div className="w-full h-full glass-panel rounded-lg overflow-hidden relative shadow-inner">
                {/* PDF Viewer frame */}
                <iframe
                  key={viewerKey} // Force React remount to trigger browser PDF viewer page updates
                  src={`${BACKEND_URL}/api/document/${encodeURIComponent(activePdf)}#page=${activePage}`}
                  className="w-full h-full border-none rounded-lg bg-slate-950"
                  title="PDF Viewer Pane"
                />
              </div>
            ) : (
              <div className="w-full h-full flex flex-col justify-center items-center text-center p-8 max-w-sm mx-auto select-none border border-dashed border-slate-850 rounded-lg bg-slate-950/20">
                <FileText size={48} className="text-slate-700 mb-4" />
                <h3 className="text-sm font-bold text-slate-400 mb-1">No Document Selected</h3>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Click on "View source page" in the chat citations or select a document from the left library sidebar.
                </p>
              </div>
            )}
          </div>

        </div>

      </div>
    </div>
  );
}
