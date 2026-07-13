import { useMutation, useQuery } from "@tanstack/react-query";
import { BarChart3, CheckCircle2, FlaskConical, Loader2, Play } from "lucide-react";
import { useState } from "react";
import { alexApi } from "./api";
import { buttonClass, ErrorNote, Field, inputClass, PageHeader, Panel } from "./AlexLayout";

export default function Evaluation() {
  const [form, setForm] = useState({
    policy_type: "isaaclab_arena_ccil.policy.ccil_bc_policy.CCILBCPolicy",
    model_path: "", meta_path: "", environment: "alex_open_microwave",
    embodiment: "alex_v2_ability_hands", num_episodes: 20, device: "cuda",
    policy_device: "cuda", video: true, camera_video: true, language_instruction: "",
  });
  const [evaluationId, setEvaluationId] = useState("");
  const evaluation = useMutation({ mutationFn: alexApi.evaluate, onSuccess: (result) => setEvaluationId(result.id) });
  const result = useQuery({
    queryKey: ["evaluation", evaluationId],
    queryFn: () => alexApi.evaluation(evaluationId),
    enabled: !!evaluationId,
    refetchInterval: (query) => ["completed", "failed", "stopped"].includes(query.state.data?.status || "") ? false : 2000,
  });
  const logs = useQuery({
    queryKey: ["evaluation-logs", evaluationId],
    queryFn: () => alexApi.evaluationLogs(evaluationId),
    enabled: !!evaluationId,
    refetchInterval: 2000,
  });
  const current = result.data || evaluation.data;
  return (
    <>
      <PageHeader eyebrow="Policy validation" title="Evaluation" description="Run a trained Alex policy against a held-out dataset and review the resulting quality metrics." />
      <div className="grid gap-6 xl:grid-cols-[.8fr_1.2fr]">
        <Panel>
          <div className="mb-6 flex items-center gap-3"><div className="rounded-xl bg-cyan-400/10 p-3"><FlaskConical className="text-cyan-300" /></div><div><h2 className="font-bold">Evaluation run</h2><p className="text-sm text-slate-500">Configure checkpoint validation</p></div></div>
          <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); evaluation.mutate(form); }}>
            <Field label="Policy type"><select className={inputClass} value={form.policy_type} onChange={(e) => setForm({ ...form, policy_type: e.target.value })}><option value="isaaclab_arena_ccil.policy.ccil_bc_policy.CCILBCPolicy">CCIL</option><option value="gr00t">GR00T server</option></select></Field>
            <Field label="Checkpoint path"><input required className={inputClass} placeholder="/datasets/alex/ccil/policy.pt" value={form.model_path} onChange={(e) => setForm({ ...form, model_path: e.target.value })} /></Field>
            <Field label="Metadata path"><input className={inputClass} placeholder="/datasets/alex/ccil/ccil_bc_meta.json" value={form.meta_path} onChange={(e) => setForm({ ...form, meta_path: e.target.value })} /></Field>
            <Field label="Task"><select className={inputClass} value={form.environment} onChange={(e) => setForm({ ...form, environment: e.target.value })}><option value="alex_open_microwave">Open microwave</option><option value="alex_open_door">Open door</option><option value="alex_lever_turn">Turn lever</option><option value="alex_put_item_in_fridge_and_close_door">Fridge task</option></select></Field>
            <Field label="Embodiment"><select className={inputClass} value={form.embodiment} onChange={(e) => setForm({ ...form, embodiment: e.target.value })}><option value="alex_v2_ability_hands">Alex V2 Ability Hands (EEF IK)</option><option value="alex_v2_ability_hands_joint_pos">Alex V2 direct joint position</option></select></Field>
            <Field label="Language instruction"><input className={inputClass} placeholder="Open the microwave" value={form.language_instruction} onChange={(e) => setForm({ ...form, language_instruction: e.target.value })} /></Field>
            <div className="grid grid-cols-2 gap-4">
              <Field label="Episodes"><input type="number" min={1} className={inputClass} value={form.num_episodes} onChange={(e) => setForm({ ...form, num_episodes: Number(e.target.value) })} /></Field>
              <Field label="Videos"><select className={inputClass} value={form.video && form.camera_video ? "all" : form.camera_video ? "camera" : "none"} onChange={(e) => setForm({ ...form, video: e.target.value === "all", camera_video: e.target.value !== "none" })}><option value="all">Viewport + policy cameras</option><option value="camera">Policy cameras</option><option value="none">None</option></select></Field>
            </div>
            <button className={`${buttonClass} w-full`} disabled={evaluation.isPending}>{evaluation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4 fill-current" />} Run evaluation</button>
          </form>
          <div className="mt-4"><ErrorNote error={evaluation.error} /></div>
        </Panel>
        <Panel className="min-h-[420px]">
          <div className="flex items-center gap-2"><BarChart3 className="h-4 w-4 text-cyan-400" /><h2 className="font-bold">Results</h2></div>
          {!current && !evaluation.isPending && <div className="grid h-[330px] place-items-center text-center"><div><div className="mx-auto grid h-16 w-16 place-items-center rounded-2xl border border-white/10 bg-white/[.025]"><BarChart3 className="text-slate-600" /></div><p className="mt-4 text-sm text-slate-500">Results will appear when evaluation completes.</p></div></div>}
          {evaluation.isPending && <div className="grid h-[330px] place-items-center"><div className="text-center"><Loader2 className="mx-auto h-8 w-8 animate-spin text-cyan-400" /><p className="mt-3 text-sm text-slate-500">Evaluating policy…</p></div></div>}
          {current && <div className="mt-6">
            <div className="mb-5 flex items-center gap-2 rounded-xl border border-emerald-400/20 bg-emerald-400/[.06] p-4 text-sm font-semibold text-emerald-300">{current.status === "running" ? <Loader2 className="h-4 w-4 animate-spin" /> : <CheckCircle2 className="h-4 w-4" />} {current.status}</div>
            {current.metrics && <div className="grid gap-3 sm:grid-cols-2">{Object.entries(current.metrics).map(([name, value]) => <div key={name} className="rounded-xl border border-white/10 bg-black/15 p-4"><div className="text-xs uppercase tracking-wider text-slate-500">{name.replace(/_/g, " ")}</div><div className="mt-2 text-2xl font-bold text-cyan-200">{typeof value === "number" ? Number(value.toFixed(4)) : value}</div></div>)}</div>}
            {!!current.artifacts?.length && <div className="mt-4 space-y-1 text-xs text-slate-400">{current.artifacts.map((artifact) => <div className="font-mono" key={artifact}>{artifact}</div>)}</div>}
            {current.error_message && <p className="mt-4 text-sm text-rose-300">{current.error_message}</p>}
            <pre className="mt-4 max-h-52 overflow-auto rounded-xl bg-black/30 p-3 text-xs text-slate-400">{Array.isArray(logs.data?.logs) ? logs.data.logs.join("\n") : logs.data?.logs || "Waiting for logs…"}</pre>
          </div>}
        </Panel>
      </div>
    </>
  );
}
