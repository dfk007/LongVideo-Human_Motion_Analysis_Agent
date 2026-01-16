"use client";

import React, { useState } from 'react';
import axios from 'axios';
import { Upload, Send, FileVideo, Clock, Play } from 'lucide-react';

// Configuration
const getApiUrl = () => {
  if (typeof window !== 'undefined') {
    // Dynamically connect to the same hostname on port 8000
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
  return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
};

interface Message {
  role: 'user' | 'agent';
  content: string;
  evidence?: any[];
}

export default function AgentInterface() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [filename, setFilename] = useState<string | null>(null);
  
  // Get API URL dynamically
  const API_URL = getApiUrl();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      console.log(`Uploading to: ${API_URL}/api/upload`);
      const res = await axios.post(`${API_URL}/api/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setFilename(res.data.filename);
      setProcessing(true); // Assuming processing starts immediately
      // In a real app, we'd poll for status. Here we just assume it's "ready" for queries soon.
      alert("Video uploaded! Processing started. Please wait a moment before querying.");
    } catch (err) {
      console.error("Upload Error:", err);
      alert(`Upload failed. Check console for details. (Target: ${API_URL})`);
    } finally {
      setUploading(false);
    }
  };

  const handleQuery = async () => {
    if (!query.trim()) return;
    
    const userMsg: Message = { role: 'user', content: query };
    setMessages(prev => [...prev, userMsg]);
    setQuery('');
    setLoading(true);

    try {
      const res = await axios.post(`${API_URL}/api/query`, { query: userMsg.content });
      const agentMsg: Message = { 
        role: 'agent', 
        content: res.data.answer,
        evidence: res.data.evidence 
      };
      setMessages(prev => [...prev, agentMsg]);
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { role: 'agent', content: "Sorry, I encountered an error." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-8">
      
      {/* Upload Section */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <FileVideo className="w-5 h-5" /> Video Ingestion
        </h2>
        
        <div className="flex gap-4 items-center">
          <input 
            type="file" 
            accept="video/*" 
            onChange={handleFileChange}
            className="block w-full text-sm text-slate-500
              file:mr-4 file:py-2 file:px-4
              file:rounded-full file:border-0
              file:text-sm file:font-semibold
              file:bg-violet-50 file:text-violet-700
              hover:file:bg-violet-100"
          />
          <button 
            onClick={handleUpload} 
            disabled={!file || uploading}
            className="bg-violet-600 text-white px-4 py-2 rounded-lg hover:bg-violet-700 disabled:opacity-50 flex gap-2 items-center"
          >
            {uploading ? 'Uploading...' : 'Upload & Process'}
            {!uploading && <Upload className="w-4 h-4" />}
          </button>
        </div>
        {filename && <p className="mt-2 text-sm text-green-600">Active Video: {filename}</p>}
      </div>

      {/* Chat Section */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 min-h-[500px] flex flex-col">
        <h2 className="text-xl font-semibold mb-4 border-b pb-2">Analysis Chat</h2>
        
        <div className="flex-1 space-y-4 overflow-y-auto mb-4 max-h-[600px]">
          {messages.map((msg, idx) => (
            <div key={idx} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
              <div className={`p-4 rounded-lg max-w-[80%] ${
                msg.role === 'user' 
                  ? 'bg-violet-600 text-white' 
                  : 'bg-slate-100 text-slate-800'
              }`}>
                <p className="whitespace-pre-wrap">{msg.content}</p>
              </div>
              
              {/* Evidence Display */}
              {msg.evidence && msg.evidence.length > 0 && (
                <div className="mt-2 ml-2 space-y-2 w-full max-w-[80%]">
                  <p className="text-xs font-bold text-slate-500 uppercase">Evidence Citations:</p>
                  {msg.evidence.map((ev, i) => (
                    <div key={i} className="bg-amber-50 border border-amber-200 p-2 rounded text-sm flex gap-2 items-center text-amber-900">
                      <Clock className="w-4 h-4 text-amber-600" />
                      <span className="font-mono font-bold">{ev.start_time}s - {ev.end_time}s</span>
                      <span className="text-xs text-amber-700/70 truncate">{ev.chunk_path}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
          {loading && <div className="text-slate-400 italic">Agent is thinking...</div>}
        </div>

        <div className="flex gap-2 border-t pt-4">
          <input 
            type="text" 
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleQuery()}
            placeholder="Ask about the video (e.g., 'Is the squat depth correct?')" 
            className="flex-1 border border-slate-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-violet-500"
          />
          <button 
            onClick={handleQuery}
            disabled={loading || !query.trim()}
            className="bg-slate-900 text-white p-3 rounded-lg hover:bg-slate-800 disabled:opacity-50"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>

    </div>
  );
}
