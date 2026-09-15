'use client'

import { useState, useEffect } from 'react'

type Character = {
  name: string
  aliases: string[]
  first_appearance: number
  description: string | null
}

type CharacterDetail = Character & {
  mentions: { chapter: number; context: string }[]
  relationships: { character: string; type: string; since_chapter: number }[]
}

export default function CharactersTab({ apiUrl, novelId, currentChapter }: {
  apiUrl: string
  novelId: number
  currentChapter: number
}) {
  const [characters, setCharacters] = useState<Character[]>([])
  const [selected, setSelected] = useState<CharacterDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [detailLoading, setDetailLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    fetch(`${apiUrl}/characters/list?novel_id=${novelId}&current_chapter=${currentChapter}`)
      .then(r => r.json())
      .then(data => {
        if (Array.isArray(data)) setCharacters(data)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [apiUrl, novelId, currentChapter])

  const loadCharacter = async (name: string) => {
    setDetailLoading(true)
    try {
      const resp = await fetch(
        `${apiUrl}/characters/${encodeURIComponent(name)}?novel_id=${novelId}&current_chapter=${currentChapter}`
      )
      const data = await resp.json()
      if (!data.error) setSelected(data)
    } catch { /* ignore */ }
    setDetailLoading(false)
  }

  if (loading) {
    return <p className="text-[#737373] text-center mt-10">Loading characters...</p>
  }

  return (
    <div className="flex flex-col md:flex-row gap-4 h-full">
      {/* Character list */}
      <div className="md:w-1/3 space-y-2">
        <h2 className="text-sm font-medium text-[#737373] mb-2">
          {characters.length} characters (up to Ch. {currentChapter})
        </h2>
        {characters.length === 0 && (
          <p className="text-sm text-[#737373]">No characters extracted yet. Ingest chapters first.</p>
        )}
        {characters.map(c => (
          <button
            key={c.name}
            onClick={() => loadCharacter(c.name)}
            className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
              selected?.name === c.name
                ? 'bg-[#3b82f6] text-white'
                : 'bg-[#1a1a1a] border border-[#2a2a2a] hover:border-[#3b82f6]'
            }`}
          >
            <p className="font-medium">{c.name}</p>
            {c.aliases.length > 0 && (
              <p className="text-xs opacity-70">aka {c.aliases.join(', ')}</p>
            )}
            <p className="text-xs opacity-50">First seen: Ch. {c.first_appearance}</p>
          </button>
        ))}
      </div>

      {/* Character detail */}
      <div className="md:w-2/3">
        {detailLoading ? (
          <p className="text-[#737373] animate-pulse">Loading...</p>
        ) : selected ? (
          <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl p-4 space-y-4">
            <div>
              <h2 className="text-lg font-bold">{selected.name}</h2>
              {selected.aliases.length > 0 && (
                <p className="text-sm text-[#737373]">Also known as: {selected.aliases.join(', ')}</p>
              )}
              <p className="text-sm mt-1">First appearance: Chapter {selected.first_appearance}</p>
              {selected.description && <p className="text-sm mt-2">{selected.description}</p>}
            </div>

            {selected.relationships.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-[#737373] mb-2">Relationships</h3>
                <div className="space-y-1">
                  {selected.relationships.map((r, i) => (
                    <div key={i} className="flex items-center gap-2 text-sm">
                      <span className="px-2 py-0.5 bg-[#2a2a2a] rounded text-xs">{r.type}</span>
                      <span>{r.character}</span>
                      <span className="text-[#737373] text-xs">since Ch. {r.since_chapter}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {selected.mentions.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-[#737373] mb-2">Appearances</h3>
                <div className="space-y-1">
                  {selected.mentions.map((m, i) => (
                    <div key={i} className="text-sm">
                      <span className="text-[#3b82f6]">Ch. {m.chapter}</span>
                      {m.context && <span className="text-[#737373]"> — {m.context}</span>}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <p className="text-[#737373] text-center mt-10">Select a character to view details</p>
        )}
      </div>
    </div>
  )
}
