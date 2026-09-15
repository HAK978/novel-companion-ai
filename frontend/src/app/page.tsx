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
  const [currentChapter, setCurrentChapter] = useState<number>(1)
  const [activeTab, setActiveTab] = useState<Tab>('Ask')

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
    fetch(`${API}/progress/${selectedNovel.id}/default`)
      .then(r => r.json())
      .then(data => {
        if (data.current_chapter) setCurrentChapter(data.current_chapter)
      })
      .catch(() => {})
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
            <span className="text-sm text-[#737373]">Ch. {currentChapter}</span>
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
        {!selectedNovel ? (
          <div className="text-center text-[#737373] mt-20">
            <p className="text-lg">No novels found</p>
            <p className="text-sm mt-2">Go to Settings to add a novel</p>
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
            {activeTab === 'Settings' && (
              <SettingsTab
                apiUrl={API} novelId={selectedNovel.id} currentChapter={currentChapter}
                setCurrentChapter={setCurrentChapter} novels={novels}
                setNovels={setNovels} setSelectedNovel={setSelectedNovel}
              />
            )}
          </>
        )}
      </main>
    </div>
  )
}
