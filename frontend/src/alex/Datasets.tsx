import { useMutation } from "@tanstack/react-query";
import { ArrowRight, CheckCircle2, Database, FileSearch, Loader2, TriangleAlert } from "lucide-react";
import { useState } from "react";
import { alexApi } from "./api";
import { buttonClass, ErrorNote, Field, inputClass, PageHeader, Panel } from "./AlexLayout";

export default function Datasets() {
  const [path, setPath] = useState("");
  const [conversion, setConversion] = useState({ source: "", destination: "", format: "gr00t" });
  const inspect = useMutation({ mutationFn: alexApi.inspectDataset });
  const convert = useMutation({ mutationFn: alexApi.convertDataset });
  return (
    <>
      <PageHeader eyebrow="Data pipeline" title="Datasets" description="Validate trajectory data before training, then convert it into the format your Alex policy expects." />
      <div className="grid gap-6 xl:grid-cols-2">
        <Panel>
          <div className="mb-6 flex items-start gap-3"><div className="rounded-xl bg-cyan-400/10 p-3"><FileSearch className="text-cyan-300" /></div><div><h2 className="font-bold">Inspect dataset</h2><p className="mt-1 text-sm text-slate-500">Check metadata, features, and warnings</p></div></div>
          <form onSubmit={(e) => { e.preventDefault(); inspect.mutate({ path }); }}>
            <Field label="Dataset path" hint="Path must be accessible from the Alex backend"><input className={inputClass} required placeholder="/data/alex/pick-and-place" value={path} onChange={(e) => setPath(e.target.value)} /></Field>
            <button className={`${buttonClass} mt-4 w-full`} disabled={inspect.isPending}>{inspect.isPending && <Loader2 className="h-4 w-4 animate-spin" />} Inspect</button>
          </form>
          <div className="mt-4"><ErrorNote error={inspect.error} /></div>
          {inspect.data && <Inspection data={inspect.data} />}
        </Panel>
        <Panel>
          <div className="mb-6 flex items-start gap-3"><div className="rounded-xl bg-indigo-400/10 p-3"><Database className="text-indigo-300" /></div><div><h2 className="font-bold">Convert dataset</h2><p className="mt-1 text-sm text-slate-500">Create a training-ready copy</p></div></div>
          <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); convert.mutate(conversion); }}>
            <Field label="Source"><input className={inputClass} required placeholder="/data/raw/session-01" value={conversion.source} onChange={(e) => setConversion({ ...conversion, source: e.target.value })} /></Field>
            <div className="flex justify-center"><ArrowRight className="h-4 w-4 rotate-90 text-slate-600" /></div>
            <Field label="Destination"><input className={inputClass} required placeholder="/data/processed/session-01" value={conversion.destination} onChange={(e) => setConversion({ ...conversion, destination: e.target.value })} /></Field>
            <Field label="Output format"><select className={inputClass} value={conversion.format} onChange={(e) => setConversion({ ...conversion, format: e.target.value })}><option value="gr00t">GR00T</option><option value="ccil">CCIL pickle</option></select></Field>
            <button className={`${buttonClass} w-full`} disabled={convert.isPending}>{convert.isPending && <Loader2 className="h-4 w-4 animate-spin" />} Prepare conversion</button>
          </form>
          <div className="mt-4"><ErrorNote error={convert.error} /></div>
          {convert.data && <div className="mt-5 rounded-xl border border-emerald-400/20 bg-emerald-400/[.06] p-4 text-sm"><div className="flex items-center gap-2 font-semibold text-emerald-300"><CheckCircle2 className="h-4 w-4" /> {convert.data.status}</div><p className="mt-2 text-slate-400">{convert.data.message || convert.data.destination}</p></div>}
        </Panel>
      </div>
    </>
  );
}

function Inspection({ data }: { data: Awaited<ReturnType<typeof alexApi.inspectDataset>> }) {
  return <div className="mt-5 rounded-xl border border-white/10 bg-black/15 p-4">
    <div className={`flex items-center gap-2 font-semibold ${data.valid ? "text-emerald-300" : "text-amber-300"}`}>{data.valid ? <CheckCircle2 className="h-4 w-4" /> : <TriangleAlert className="h-4 w-4" />}{data.valid ? "Dataset is valid" : "Dataset needs attention"}</div>
    <div className="mt-4 grid grid-cols-2 gap-3 text-sm">{[["Format", data.format], ["Episodes", data.episodes], ["Frames", data.frames], ["Size", data.size_bytes ? `${(data.size_bytes / 1e9).toFixed(2)} GB` : undefined]].map(([label, value]) => <div key={label as string} className="rounded-lg bg-white/[.035] p-3"><div className="text-xs text-slate-500">{label}</div><div className="mt-1 font-semibold">{value ?? "—"}</div></div>)}</div>
    {!!data.features?.length && <div className="mt-4 flex flex-wrap gap-2">{data.features.map((feature) => <span key={feature} className="rounded-full bg-cyan-400/10 px-2.5 py-1 text-xs text-cyan-200">{feature}</span>)}</div>}
    {!!data.warnings?.length && <ul className="mt-4 space-y-1 text-xs text-amber-300">{data.warnings.map((warning) => <li key={warning}>• {warning}</li>)}</ul>}
  </div>;
}
