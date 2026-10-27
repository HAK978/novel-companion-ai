'use client'

import { useState } from 'react'

type Source = {
  text: string
  chapter_number: number
  chapter_title: string
  relevance_score: number
}

type Message = {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  // the reading position it was asked at: after moving back, later answers are hidden
  chapter: number
}

export default function AskTab({ apiUrl, novelId, currentChapter }: {
  apiUrl: string
  novelId: number
  currentChapter: number
}) {
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)

  const handleAsk = async () => {
    if (!query.trim() || loading) return

    const askedAt = currentChapter
    const userMsg: Message = { role: 'user', content: query, chapter: askedAt }
    setMessages(prev => [...prev, userMsg])
    setQuery('')
    setLoading(true)

    try {
      const resp = await fetch(`${apiUrl}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          current_chapter: askedAt,
          novel_id: novelId,
        }),
      })
      if (!resp.ok) {
        // an error is not an answer: it used to show as "No answer generated."
        setMessages(prev => [...prev, { role: 'assistant', chapter: askedAt,
          content: `Something went wrong answering that (HTTP ${resp.status}).` }])
      } else {
        const data = await resp.json()
        const assistantMsg: Message = {
          role: 'assistant',
          content: data.answer || 'No answer generated.',
          sources: data.sources || [],
          chapter: askedAt,
        }
        setMessages(prev => [...prev, assistantMsg])
      }
    } catch {
      setMessages(prev => [...prev, { role: 'assistant', chapter: askedAt,
        content: 'Error connecting to the server.' }])
    }

    setLoading(false)
  }

  const visible = messages.filter(m => m.chapter <= currentChapter)

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 space-y-4 mb-4">
        {visible.length === 0 && (
          <div className="text-center text-[#737373] mt-16">
            <p className="text-lg">Ask anything about the novel</p>
            <p className="text-sm mt-2">Only chapters 1 to {currentChapter} are searched for answers</p>
          </div>
        )}
        {visible.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] rounded-2xl px-4 py-3 ${
              msg.role === 'user'
                ? 'bg-[#3b82f6] text-white'
                : 'bg-[#1a1a1a] border border-[#2a2a2a]'
            }`}>
              <p className="whitespace-pre-wrap text-sm">{msg.content}</p>
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-3 pt-2 border-t border-[#2a2a2a]">
                  <p className="text-xs text-[#737373] mb-1">Sources:</p>
                  {msg.sources.slice(0, 3).map((s, j) => (
                    <p key={j} className="text-xs text-[#737373]">
                      Ch.{s.chapter_number}: {s.chapter_title}
                    </p>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-2xl px-4 py-3">
              <p className="text-sm text-[#737373] animate-pulse">Thinking...</p>
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="sticky bottom-0 bg-[#0a0a0a] pt-2 pb-4">
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleAsk()}
            placeholder="Ask about the novel..."
            className="flex-1 bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-[#3b82f6]"
          />
          <button
            onClick={handleAsk}
            disabled={loading || !query.trim()}
            className="bg-[#3b82f6] text-white px-5 py-3 rounded-xl text-sm font-medium disabled:opacity-50 hover:bg-[#2563eb] transition-colors"
          >
            Ask
          </button>
        </div>
      </div>
    </div>
  )
}
