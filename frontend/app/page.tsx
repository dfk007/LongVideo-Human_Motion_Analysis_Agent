import AgentInterface from "./components/AgentInterface";

export default function Home() {
  return (
    <main className="min-h-screen bg-slate-50 py-8">
      <div className="max-w-4xl mx-auto px-6 mb-8">
        <h1 className="text-3xl font-bold text-slate-900">Motion Analysis Agent</h1>
        <p className="text-slate-500">Long-form video understanding powered by Gemini 2.0</p>
      </div>

      <AgentInterface />
    </main>
  );
}