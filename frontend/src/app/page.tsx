'use client'

import { useState, useEffect } from 'react'
import AskTab from '@/components/AskTab'
import CharactersTab from '@/components/CharactersTab'
import SummaryTab from '@/components/SummaryTab'
import SettingsTab from '@/components/SettingsTab'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

type Novel = {
  id: number
  title: string
  author: string | null
  total_chapters: number
}

const tabs = ['Ask', 'Characters', 'Summary', 'Settings'] as const
type Tab = typeof tabs[number]

export default function Home() {
  const [novels, setNovels] = useState<Novel[]>([])
  const [selectedNovel, setSelectedNovel] = useState<Novel | null>(null)
  // Progress is tracked per novel. A single shared value carried one novel's
  // chapter over to the next on switch, and kept it if the new novel had none.
  const [progress, setProgress] = useState<Record<number, number>>({})
  const [activeTab, setActiveTab] = useState<Tab>('Ask')

  const currentChapter = selectedNovel ? progress[selectedNovel.id] ?? 1 : 1

  const setCurrentChapter = (chapter: number) => {
    if (selectedNovel) {
      setProgress(prev => ({ ...prev, [selectedNovel.id]: chapter }))
    }
  }

  useEffect(() => {
    fetch(`${API}/novels`)
      .then(r => r.json())
      .then(data => {
        setNovels(data)
        if (data.length > 0) setSelectedNovel(data[0])
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (!selectedNovel) return
    const novelId = selectedNovel.id
    let cancelled = false

    fetch(`${API}/progress/${novelId}/default`)
      .then(r => r.json())
      .then(data => {
        if (!cancelled && data.current_chapter) {
          setProgress(prev => ({ ...prev, [novelId]: data.current_chapter }))
        }
      })
      .catch(() => {})

    return () => {
      cancelled = true
    }
  }, [selectedNovel])

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto">
      <header className="p-4 border-b border-[#2a2a2a]">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-bold">Novel Companion AI</h1>
          <div className="flex items-center gap-3">
            <select
              className="bg-[#1a1a1a] border border-[#2a2a2a] rounded px-3 py-1.5 text-sm"
              value={selectedNovel?.id || ''}
              onChange={(e) => {
                const novel = novels.find(n => n.id === Number(e.target.value))
                if (novel) setSelectedNovel(novel)
              }}
            >
              {novels.map(n => (
                <option key={n.id} value={n.id}>{n.title}</option>
              ))}
              {novels.length === 0 && <option value="">No novels</option>}
            </select>
            {selectedNovel && (
              <span className="text-sm text-[#737373]">Ch. {currentChapter}</span>
            )}
          </div>
        </div>
        <nav className="flex gap-1 mt-3">
          {tabs.map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 text-sm rounded-lg transition-colors ${
                activeTab === tab
                  ? 'bg-[#3b82f6] text-white'
                  : 'text-[#737373] hover:text-white hover:bg-[#1a1a1a]'
              }`}
            >
              {tab}
            </button>
          ))}
        </nav>
      </header>

      <main className="flex-1 overflow-y-auto p-4">
        {/* Settings renders with or without a novel: it is where the first one
            gets added, so gating it behind a selection was a dead end. */}
        {activeTab === 'Settings' ? (
          <SettingsTab
            apiUrl={API} novelId={selectedNovel?.id ?? null} currentChapter={currentChapter}
            setCurrentChapter={setCurrentChapter} novels={novels}
            setNovels={setNovels} setSelectedNovel={setSelectedNovel}
          />
        ) : !selectedNovel ? (
          <div className="text-center text-[#737373] mt-20">
            <p className="text-lg">No novels found</p>
            <button
              onClick={() => setActiveTab('Settings')}
              className="text-sm mt-2 text-[#3b82f6] hover:underline"
            >
              Add a novel in Settings
            </button>
          </div>
        ) : (
          <>
            {activeTab === 'Ask' && (
              <AskTab apiUrl={API} novelId={selectedNovel.id} currentChapter={currentChapter} />
            )}
            {activeTab === 'Characters' && (
              <CharactersTab apiUrl={API} novelId={selectedNovel.id} currentChapter={currentChapter} />
            )}
            {activeTab === 'Summary' && (
              <SummaryTab apiUrl={API} novelId={selectedNovel.id} currentChapter={currentChapter} />
            )}
          </>
        )}
      </main>
    </div>
  )
}
