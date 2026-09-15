'use client'

import { useState } from 'react'

type Novel = {
  id: number
  title: string
  author: string | null
  total_chapters: number
}

export default function SettingsTab({ apiUrl, novelId, currentChapter, setCurrentChapter, novels, setNovels, setSelectedNovel }: {
  apiUrl: string
  novelId: number
  currentChapter: number
  setCurrentChapter: (ch: number) => void
  novels: Novel[]
  setNovels: (novels: Novel[]) => void
  setSelectedNovel: (novel: Novel | null) => void
}) {
  const [chapter, setChapter] = useState(currentChapter)
  const [saving, setSaving] = useState(false)
  const [newTitle, setNewTitle] = useState('')
  const [newAuthor, setNewAuthor] = useState('')
  const [creating, setCreating] = useState(false)
  const [ingestPath, setIngestPath] = useState('')
  const [ingestMax, setIngestMax] = useState<number | ''>('')
  const [ingesting, setIngesting] = useState(false)
  const [ingestStatus, setIngestStatus] = useState('')
  const [health, setHealth] = useState<any>(null)

  const saveProgress = async () => {
    setSaving(true)
    try {
      await fetch(`${apiUrl}/progress`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ novel_id: novelId, current_chapter: chapter, user_id: 'default' }),
      })
      setCurrentChapter(chapter)
    } catch { /* ignore */ }
    setSaving(false)
  }

  const createNovel = async () => {
    if (!newTitle.trim()) return
    setCreating(true)
    try {
      const resp = await fetch(`${apiUrl}/novels`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: newTitle, author: newAuthor || null }),
      })
      const data = await resp.json()
      const novel: Novel = { id: data.id, title: newTitle, author: newAuthor || null, total_chapters: 0 }
      setNovels([...novels, novel])
      setSelectedNovel(novel)
      setNewTitle('')
      setNewAuthor('')
    } catch { /* ignore */ }
    setCreating(false)
  }

  const ingestFromSource = async () => {
    if (!ingestPath.trim()) return
    setIngesting(true)
    setIngestStatus('Submitting...')
    try {
      const resp = await fetch(`${apiUrl}/ingest/from-source`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          novel_id: novelId,
          source_type: 'local_json',
          source_path: ingestPath,
          max_chapters: ingestMax || null,
        }),
      })
      const data = await resp.json()
      if (data.task_id) {
        setIngestStatus(`Queued (task: ${data.task_id.slice(0, 8)}...)`)
        // Poll for completion
        const poll = setInterval(async () => {
          const s = await fetch(`${apiUrl}/ingest/status/${data.task_id}`).then(r => r.json())
          if (s.status === 'SUCCESS') {
            setIngestStatus(`Done! ${s.result?.chapters_succeeded || 0} chapters ingested.`)
            setIngesting(false)
            clearInterval(poll)
          } else if (s.status === 'FAILURE') {
            setIngestStatus('Failed.')
            setIngesting(false)
            clearInterval(poll)
          } else {
            setIngestStatus(`Processing... (${s.status})`)
          }
        }, 5000)
      }
    } catch {
      setIngestStatus('Error connecting to server.')
      setIngesting(false)
    }
  }

  const checkHealth = async () => {
    try {
      const resp = await fetch(`${apiUrl}/health`)
      setHealth(await resp.json())
    } catch {
      setHealth({ status: 'unreachable' })
    }
  }

  return (
    <div className="space-y-6">
      {/* Reading Progress */}
      <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl p-4">
        <h2 className="text-lg font-bold mb-2">Reading Progress</h2>
        <p className="text-sm text-[#737373] mb-3">Set your current chapter to control spoiler filtering</p>
        <div className="flex items-center gap-3">
          <input
            type="number"
            value={chapter}
            onChange={(e) => setChapter(Number(e.target.value))}
            min={1}
            className="w-28 bg-[#0a0a0a] border border-[#2a2a2a] rounded px-3 py-1.5 text-sm"
          />
          <button
            onClick={saveProgress}
            disabled={saving}
            className="bg-[#3b82f6] text-white px-4 py-1.5 rounded-lg text-sm disabled:opacity-50"
          >
            {saving ? 'Saving...' : 'Save'}
          </button>
          <span className="text-xs text-[#737373]">Currently: Ch. {currentChapter}</span>
        </div>
      </div>

      {/* Add Novel */}
      <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl p-4">
        <h2 className="text-lg font-bold mb-2">Add Novel</h2>
        <div className="flex flex-col gap-2 mb-3">
          <input
            type="text"
            value={newTitle}
            onChange={(e) => setNewTitle(e.target.value)}
            placeholder="Novel title"
            className="bg-[#0a0a0a] border border-[#2a2a2a] rounded px-3 py-1.5 text-sm"
          />
          <input
            type="text"
            value={newAuthor}
            onChange={(e) => setNewAuthor(e.target.value)}
            placeholder="Author (optional)"
            className="bg-[#0a0a0a] border border-[#2a2a2a] rounded px-3 py-1.5 text-sm"
          />
        </div>
        <button
          onClick={createNovel}
          disabled={creating || !newTitle.trim()}
          className="bg-[#3b82f6] text-white px-4 py-1.5 rounded-lg text-sm disabled:opacity-50"
        >
          {creating ? 'Creating...' : 'Create Novel'}
        </button>
      </div>

      {/* Ingest Chapters */}
      <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl p-4">
        <h2 className="text-lg font-bold mb-2">Ingest Chapters</h2>
        <p className="text-sm text-[#737373] mb-3">Path to chapter files on the server</p>
        <div className="flex flex-col gap-2 mb-3">
          <input
            type="text"
            value={ingestPath}
            onChange={(e) => setIngestPath(e.target.value)}
            placeholder="/path/to/chapters or /path/to/file.json"
            className="bg-[#0a0a0a] border border-[#2a2a2a] rounded px-3 py-1.5 text-sm"
          />
          <input
            type="number"
            value={ingestMax}
            onChange={(e) => setIngestMax(e.target.value ? Number(e.target.value) : '')}
            placeholder="Max chapters (optional)"
            className="w-48 bg-[#0a0a0a] border border-[#2a2a2a] rounded px-3 py-1.5 text-sm"
          />
        </div>
        <button
          onClick={ingestFromSource}
          disabled={ingesting || !ingestPath.trim()}
          className="bg-[#3b82f6] text-white px-4 py-1.5 rounded-lg text-sm disabled:opacity-50"
        >
          {ingesting ? 'Ingesting...' : 'Start Ingestion'}
        </button>
        {ingestStatus && <p className="text-sm text-[#737373] mt-2">{ingestStatus}</p>}
      </div>

      {/* System Health */}
      <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl p-4">
        <h2 className="text-lg font-bold mb-2">System Health</h2>
        <button
          onClick={checkHealth}
          className="bg-[#2a2a2a] text-white px-4 py-1.5 rounded-lg text-sm hover:bg-[#3a3a3a] transition-colors"
        >
          Check Health
        </button>
        {health && (
          <div className="mt-3 space-y-1">
            <p className={`text-sm font-medium ${health.status === 'ok' ? 'text-green-400' : 'text-red-400'}`}>
              Status: {health.status}
            </p>
            {health.services && Object.entries(health.services).map(([name, info]: [string, any]) => (
              <p key={name} className="text-xs text-[#737373]">
                <span className={info.status === 'ok' ? 'text-green-400' : 'text-red-400'}>●</span>
                {' '}{name}: {info.status}
                {info.chunks_indexed !== undefined && ` (${info.chunks_indexed} chunks)`}
                {info.workers !== undefined && ` (${info.workers} workers)`}
                {info.model && ` (${info.model})`}
              </p>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
