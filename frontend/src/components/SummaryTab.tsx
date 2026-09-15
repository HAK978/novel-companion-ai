'use client'

import { useState } from 'react'

export default function SummaryTab({ apiUrl, novelId, currentChapter }: {
  apiUrl: string
  novelId: number
  currentChapter: number
}) {
  const [startChapter, setStartChapter] = useState(1)
  const [endChapter, setEndChapter] = useState(currentChapter)
  const [summary, setSummary] = useState('')
  const [cached, setCached] = useState(false)
  const [loading, setLoading] = useState(false)
  const [catchUpResult, setCatchUpResult] = useState<any>(null)
  const [catchUpLoading, setCatchUpLoading] = useState(false)

  const handleSummarize = async () => {
    setLoading(true)
    setSummary('')
    try {
      const resp = await fetch(`${apiUrl}/summarize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          novel_id: novelId,
          start_chapter: startChapter,
          end_chapter: endChapter,
          current_chapter: currentChapter,
        }),
      })
      const data = await resp.json()
      setSummary(data.summary || data.error || 'No summary generated.')
      setCached(data.cached || false)
    } catch {
      setSummary('Error connecting to the server.')
    }
    setLoading(false)
  }

  const handleCatchUp = async () => {
    setCatchUpLoading(true)
    setCatchUpResult(null)
    try {
      const resp = await fetch(`${apiUrl}/catch-me-up`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ novel_id: novelId, user_id: 'default' }),
      })
      const data = await resp.json()
      setCatchUpResult(data)
    } catch {
      setCatchUpResult({ error: 'Error connecting to the server.' })
    }
    setCatchUpLoading(false)
  }

  return (
    <div className="space-y-6">
      {/* Catch Me Up */}
      <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl p-4">
        <h2 className="text-lg font-bold mb-2">Catch Me Up</h2>
        <p className="text-sm text-[#737373] mb-3">
          Get a summary of recent chapters based on your reading position (Ch. {currentChapter})
        </p>
        <button
          onClick={handleCatchUp}
          disabled={catchUpLoading}
          className="bg-[#3b82f6] text-white px-5 py-2 rounded-lg text-sm font-medium disabled:opacity-50 hover:bg-[#2563eb] transition-colors"
        >
          {catchUpLoading ? 'Generating...' : 'Catch Me Up'}
        </button>

        {catchUpResult && (
          <div className="mt-4 space-y-3">
            {catchUpResult.error ? (
              <p className="text-red-400 text-sm">{catchUpResult.error}</p>
            ) : (
              <>
                <div>
                  <p className="text-xs text-[#737373]">
                    Chapters {catchUpResult.summary_range?.start} - {catchUpResult.summary_range?.end}
                    {catchUpResult.cached && ' (cached)'}
                  </p>
                  <p className="text-sm mt-2 whitespace-pre-wrap">{catchUpResult.summary}</p>
                </div>
                {catchUpResult.new_characters?.length > 0 && (
                  <div>
                    <p className="text-xs text-[#737373] mb-1">New characters in this range:</p>
                    {catchUpResult.new_characters.map((c: any, i: number) => (
                      <p key={i} className="text-sm">
                        <span className="text-[#3b82f6]">{c.name}</span>
                        {c.description && <span className="text-[#737373]"> — {c.description}</span>}
                      </p>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>

      {/* Custom Summary */}
      <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl p-4">
        <h2 className="text-lg font-bold mb-2">Custom Summary</h2>
        <p className="text-sm text-[#737373] mb-3">Summarize a specific chapter range</p>
        <div className="flex items-center gap-3 mb-3">
          <div>
            <label className="text-xs text-[#737373]">From</label>
            <input
              type="number"
              value={startChapter}
              onChange={(e) => setStartChapter(Number(e.target.value))}
              min={1}
              max={currentChapter}
              className="block w-24 bg-[#0a0a0a] border border-[#2a2a2a] rounded px-3 py-1.5 text-sm mt-1"
            />
          </div>
          <div>
            <label className="text-xs text-[#737373]">To</label>
            <input
              type="number"
              value={endChapter}
              onChange={(e) => setEndChapter(Number(e.target.value))}
              min={1}
              max={currentChapter}
              className="block w-24 bg-[#0a0a0a] border border-[#2a2a2a] rounded px-3 py-1.5 text-sm mt-1"
            />
          </div>
          <button
            onClick={handleSummarize}
            disabled={loading}
            className="bg-[#3b82f6] text-white px-5 py-2 rounded-lg text-sm font-medium disabled:opacity-50 hover:bg-[#2563eb] transition-colors mt-5"
          >
            {loading ? 'Generating...' : 'Summarize'}
          </button>
        </div>

        {summary && (
          <div className="mt-3">
            {cached && <p className="text-xs text-[#737373] mb-1">(cached)</p>}
            <p className="text-sm whitespace-pre-wrap">{summary}</p>
          </div>
        )}
      </div>
    </div>
  )
}
