import { useMutation, useQuery } from "@tanstack/react-query";
import { AlertTriangle, BarChart3, CheckCircle2, FlaskConical, Loader2, Play, Square } from "lucide-react";
import { useMemo, useState } from "react";
import { alexApi, type DatasetEvalRequest, type RolloutRequest } from "./api";
import { buttonClass, ErrorNote, Field, inputClass, PageHeader, Panel, secondaryButtonClass } from "./AlexLayout";

const terminal = new Set(["completed", "failed", "stopped", "blocked", "interrupted"]);

export default function Evaluation() {
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: alexApi.jobs });
  const completedJobs = useMemo(() => (jobs.data || []).filter((job) => job.status === "completed"), [jobs.data]);
  const [source, setSource] = useState<"job" | "policy">("job");
  const [form, setForm] = useState<RolloutRequest>({
    target: "arena",
    inference_location: "remote",
    job_id: "",
    policy_ref: "",
    checkpoint: "latest",
    dataset_repo_id: "",
    gpu: "0",
    task: "Turn the lever",
    environment: "alex_empty",
    embodiment: "alex_v2_ability_hands",
    usd: "isaaclab_arena/assets/lever_sim/LEVER_AGAIN.usd",
    num_episodes: 20,
    video: true,
    camera_video: true,
  });
  const [rolloutId, setRolloutId] = useState("");
  const [datasetEvalSource, setDatasetEvalSource] = useState<"job" | "policy">("job");
  const [datasetEvalForm, setDatasetEvalForm] = useState({
    job_id: "",
    policy_ref: "",
    checkpoint: "latest",
    dataset_repo_id: "",
    dataset_revision: "",
    episode_spec: "0:14,29:50,115:120",
    gpu: "0",
    batch_size: 8,
    num_workers: 4,
    policy_use_bf16: true,
  });
  const [datasetEvalError, setDatasetEvalError] = useState<string | null>(null);
  const [datasetEvalId, setDatasetEvalId] = useState("");
  const launch = useMutation({
    mutationFn: (request: RolloutRequest) => alexApi.rollout(request),
    onSuccess: (result) => setRolloutId(result.id),
  });
  const result = useQuery({
    queryKey: ["rollout", rolloutId], queryFn: () => alexApi.rolloutStatus(rolloutId), enabled: !!rolloutId,
    refetchInterval: (query) => terminal.has(query.state.data?.status || "") ? false : 2000,
  });
  const logs = useQuery({
    queryKey: ["rollout-logs", rolloutId], queryFn: () => alexApi.rolloutLogs(rolloutId), enabled: !!rolloutId,
    refetchInterval: (query) => terminal.has(result.data?.status || "") ? false : 2000,
  });
  const stop = useMutation({ mutationFn: () => alexApi.stopRollout(rolloutId), onSuccess: () => result.refetch() });
  const current = result.data || launch.data;
  const launchDatasetEval = useMutation({
    mutationFn: (request: DatasetEvalRequest) => alexApi.datasetEval(request),
    onSuccess: (evalResult) => setDatasetEvalId(evalResult.id),
  });
  const datasetEvalResult = useQuery({
    queryKey: ["dataset-eval", datasetEvalId],
    queryFn: () => alexApi.datasetEvalStatus(datasetEvalId),
    enabled: !!datasetEvalId,
    refetchInterval: (query) => terminal.has(query.state.data?.status || "") ? false : 2000,
  });
  const datasetEvalLogs = useQuery({
    queryKey: ["dataset-eval-logs", datasetEvalId],
    queryFn: () => alexApi.datasetEvalLogs(datasetEvalId),
    enabled: !!datasetEvalId,
    refetchInterval: () => terminal.has(datasetEvalResult.data?.status || "") ? false : 2000,
  });
  const stopDatasetEval = useMutation({
    mutationFn: () => alexApi.stopDatasetEval(datasetEvalId),
    onSuccess: () => datasetEvalResult.refetch(),
  });
  const currentDatasetEval = datasetEvalResult.data || launchDatasetEval.data;

  function setTarget(target: RolloutRequest["target"]) {
    if (target === "arena") {
      setForm({
        ...form,
        target,
        environment: "alex_empty",
        usd: form.usd || "isaaclab_arena/assets/lever_sim/LEVER_AGAIN.usd",
      });
      return;
    }
    if (target === "sim") {
      setForm({
        ...form,
        target,
        environment: form.environment.startsWith("Isaac-") ? form.environment : "Isaac-Alex-Lever-Play-v0",
      });
      return;
    }
    setForm({ ...form, target });
  }

  function submit() {
    const request = { ...form };
    if (source === "job") {
      delete request.policy_ref;
      delete request.dataset_repo_id;
    } else {
      delete request.job_id;
    }
    launch.mutate(request);
  }

  function submitDatasetEval() {
    let datasetEpisodes: number[];
    try {
      datasetEpisodes = parseEpisodeSpec(datasetEvalForm.episode_spec);
      setDatasetEvalError(null);
    } catch (error) {
      setDatasetEvalError(error instanceof Error ? error.message : "Invalid episode selection");
      return;
    }
    const request: DatasetEvalRequest = {
      checkpoint: datasetEvalForm.checkpoint,
      dataset_episodes: datasetEpisodes,
      gpu: datasetEvalForm.gpu,
      batch_size: datasetEvalForm.batch_size,
      num_workers: datasetEvalForm.num_workers,
      policy_use_bf16: datasetEvalForm.policy_use_bf16,
    };
    if (datasetEvalForm.dataset_revision) request.dataset_revision = datasetEvalForm.dataset_revision;
    if (datasetEvalSource === "job") {
      request.job_id = datasetEvalForm.job_id;
    } else {
      request.policy_ref = datasetEvalForm.policy_ref;
      request.dataset_repo_id = datasetEvalForm.dataset_repo_id;
    }
    launchDatasetEval.mutate(request);
  }

  return (
    <>
      <PageHeader eyebrow="Policy deployment" title="Rollout" description="Load a trained LeRobot policy on the GPU host, then run evaluation episodes in Isaac Lab Arena (or legacy Isaac Lab) / inspect physical-Alex deployment readiness." />
      <div className="grid gap-6 xl:grid-cols-[.85fr_1.15fr]">
        <Panel>
          <div className="mb-6 flex items-center gap-3"><div className="rounded-xl bg-cyan-400/10 p-3"><FlaskConical className="text-cyan-300" /></div><div><h2 className="font-bold">New rollout</h2><p className="text-sm text-slate-500">Checkpoint, target, and task</p></div></div>
          <form className="space-y-4" onSubmit={(event) => { event.preventDefault(); submit(); }}>
            <Field label="Target">
              <select className={inputClass} value={form.target} onChange={(e) => setTarget(e.target.value as RolloutRequest["target"])}>
                <option value="arena">Isaac Lab Arena</option>
                <option value="sim">Isaac Lab / Isaac Sim (legacy)</option>
                <option value="robot">Physical Alex via RDX</option>
              </select>
            </Field>
            <Field label="Checkpoint source"><select className={inputClass} value={source} onChange={(e) => setSource(e.target.value as "job" | "policy")}><option value="job">Completed LeLab job</option><option value="policy">Hub or local policy ref</option></select></Field>
            {source === "job" ? <Field label="Training job"><select required className={inputClass} value={form.job_id} onChange={(e) => setForm({ ...form, job_id: e.target.value })}><option value="">Select a completed job</option>{completedJobs.map((job) => <option value={job.id} key={job.id}>{job.name || job.id}</option>)}</select></Field> : <><Field label="Policy ref"><input required className={inputClass} placeholder="owner/model or /local/pretrained_model" value={form.policy_ref} onChange={(e) => setForm({ ...form, policy_ref: e.target.value })} /></Field><Field label="Dataset repo" hint="Required when the policy has no LeLab deployment manifest"><input required className={inputClass} placeholder="owner/alex-dataset" value={form.dataset_repo_id} onChange={(e) => setForm({ ...form, dataset_repo_id: e.target.value })} /></Field></>}
            <Field label="Policy server">
              <select className={inputClass} value={form.inference_location || "remote"} onChange={(e) => setForm({ ...form, inference_location: e.target.value as RolloutRequest["inference_location"] })}>
                <option value="remote">SSH training host</option>
                <option value="local">Local Docker</option>
              </select>
            </Field>
            <div className="grid grid-cols-2 gap-4"><Field label="Checkpoint"><input className={inputClass} value={form.checkpoint} onChange={(e) => setForm({ ...form, checkpoint: e.target.value })} /></Field><Field label="Inference GPU"><input className={inputClass} value={form.gpu} onChange={(e) => setForm({ ...form, gpu: e.target.value })} /></Field></div>
            <Field label="Instruction"><input className={inputClass} value={form.task} onChange={(e) => setForm({ ...form, task: e.target.value })} /></Field>
            {form.target === "arena" ? (
              <>
                <Field label="Arena environment">
                  <select className={inputClass} value={form.environment} onChange={(e) => setForm({ ...form, environment: e.target.value })}>
                    <option value="alex_empty">alex_empty</option>
                    <option value="alex_lever_turn">alex_lever_turn</option>
                  </select>
                </Field>
                <Field label="USD asset">
                  <input className={inputClass} value={form.usd || ""} onChange={(e) => setForm({ ...form, usd: e.target.value })} />
                </Field>
                <div className="grid grid-cols-2 gap-4">
                  <Field label="Episodes"><input type="number" min={1} className={inputClass} value={form.num_episodes} onChange={(e) => setForm({ ...form, num_episodes: Number(e.target.value) })} /></Field>
                  <Field label="Videos">
                    <select className={inputClass} value={form.video && form.camera_video ? "all" : form.camera_video ? "camera" : "none"} onChange={(e) => setForm({ ...form, video: e.target.value === "all", camera_video: e.target.value !== "none" })}>
                      <option value="all">Viewport + cameras</option>
                      <option value="camera">Policy cameras</option>
                      <option value="none">None</option>
                    </select>
                  </Field>
                </div>
              </>
            ) : form.target === "sim" ? (
              <>
                <Field label="Isaac Lab task">
                  <select className={inputClass} value={form.environment} onChange={(e) => setForm({ ...form, environment: e.target.value })}>
                    <option value="Isaac-Alex-Lever-Play-v0">Alex lever</option>
                    <option value="Isaac-Standing-Alex-Play-v0">Standing Alex</option>
                    <option value="Isaac-WalkingFlat-Alex-Play-v0">Walking flat Alex</option>
                    <option value="Isaac-WalkingUneven-Alex-Play-v0">Walking uneven Alex</option>
                  </select>
                </Field>
                <div className="grid grid-cols-2 gap-4">
                  <Field label="Episodes"><input type="number" min={1} className={inputClass} value={form.num_episodes} onChange={(e) => setForm({ ...form, num_episodes: Number(e.target.value) })} /></Field>
                  <Field label="Videos">
                    <select className={inputClass} value={form.video && form.camera_video ? "all" : form.camera_video ? "camera" : "none"} onChange={(e) => setForm({ ...form, video: e.target.value === "all", camera_video: e.target.value !== "none" })}>
                      <option value="all">Viewport + cameras</option>
                      <option value="camera">Policy cameras</option>
                      <option value="none">None</option>
                    </select>
                  </Field>
                </div>
              </>
            ) : (
              <div className="rounded-xl border border-amber-400/20 bg-amber-400/[.07] p-4 text-sm text-amber-200">
                <div className="flex items-center gap-2 font-semibold"><AlertTriangle className="h-4 w-4" /> Safety-gated target</div>
                <p className="mt-2 text-xs leading-5 text-amber-100/70">This checks the Alex hardware capability contract and refuses motion until state readback, frame transforms, complete action sinks, and the watchdog are available.</p>
              </div>
            )}
            <button className={`${buttonClass} w-full`} disabled={launch.isPending}>
              {launch.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4 fill-current" />}
              {form.target === "robot" ? "Check robot readiness" : "Run rollout"}
            </button>
          </form>
          <div className="mt-4"><ErrorNote error={launch.error || stop.error} /></div>
        </Panel>
        <Panel className="min-h-[500px]">
          <div className="flex items-center justify-between"><div className="flex items-center gap-2"><BarChart3 className="h-4 w-4 text-cyan-400" /><h2 className="font-bold">Run status</h2></div>{current?.status === "running" && <button className={secondaryButtonClass} onClick={() => stop.mutate()} disabled={stop.isPending}><Square className="h-4 w-4" /> Stop</button>}</div>
          {!current && !launch.isPending && <div className="grid h-[390px] place-items-center text-center text-sm text-slate-500">Choose a checkpoint and target to begin.</div>}
          {launch.isPending && <div className="grid h-[390px] place-items-center"><div className="text-center"><Loader2 className="mx-auto h-8 w-8 animate-spin text-cyan-400" /><p className="mt-3 text-sm text-slate-500">Starting remote inference…</p></div></div>}
          {current && <div className="mt-6 space-y-4"><div className={`flex items-center gap-2 rounded-xl border p-4 text-sm font-semibold ${current.status === "blocked" || current.status === "failed" ? "border-amber-400/20 bg-amber-400/[.06] text-amber-300" : "border-emerald-400/20 bg-emerald-400/[.06] text-emerald-300"}`}>{terminal.has(current.status) ? <CheckCircle2 className="h-4 w-4" /> : <Loader2 className="h-4 w-4 animate-spin" />} {current.status}</div>{current.policy_ref && <div className="break-all rounded-xl bg-black/20 p-3 font-mono text-xs text-slate-400">{current.policy_ref}</div>}{current.blockers?.map((blocker) => <div className="flex gap-2 text-sm text-amber-200" key={blocker}><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />{blocker}</div>)}{current.error_message && <p className="text-sm text-rose-300">{current.error_message}</p>}<pre className="max-h-64 overflow-auto rounded-xl bg-black/30 p-3 text-xs text-slate-400">{Array.isArray(logs.data?.logs) ? logs.data.logs.join("\n") : logs.data?.logs || "Waiting for logs…"}</pre></div>}
        </Panel>
      </div>
      <div className="mt-6 grid gap-6 xl:grid-cols-[.85fr_1.15fr]">
        <Panel>
          <div className="mb-6 flex items-center gap-3"><div className="rounded-xl bg-emerald-400/10 p-3"><BarChart3 className="text-emerald-300" /></div><div><h2 className="font-bold">Held-out dataset eval</h2><p className="text-sm text-slate-500">Compute policy loss on selected LeRobot episodes</p></div></div>
          <form className="space-y-4" onSubmit={(event) => { event.preventDefault(); submitDatasetEval(); }}>
            <Field label="Checkpoint source"><select className={inputClass} value={datasetEvalSource} onChange={(e) => setDatasetEvalSource(e.target.value as "job" | "policy")}><option value="job">Completed LeLab job</option><option value="policy">Hub policy ref</option></select></Field>
            {datasetEvalSource === "job" ? (
              <Field label="Training job"><select required className={inputClass} value={datasetEvalForm.job_id} onChange={(e) => setDatasetEvalForm({ ...datasetEvalForm, job_id: e.target.value })}><option value="">Select a completed job</option>{completedJobs.map((job) => <option value={job.id} key={job.id}>{job.name || job.id}</option>)}</select></Field>
            ) : (
              <>
                <Field label="Policy ref"><input required className={inputClass} placeholder="owner/model" value={datasetEvalForm.policy_ref} onChange={(e) => setDatasetEvalForm({ ...datasetEvalForm, policy_ref: e.target.value })} /></Field>
                <Field label="Dataset repo"><input required className={inputClass} placeholder="owner/alex-dataset" value={datasetEvalForm.dataset_repo_id} onChange={(e) => setDatasetEvalForm({ ...datasetEvalForm, dataset_repo_id: e.target.value })} /></Field>
              </>
            )}
            <div className="grid grid-cols-2 gap-4">
              <Field label="Checkpoint"><input className={inputClass} value={datasetEvalForm.checkpoint} onChange={(e) => setDatasetEvalForm({ ...datasetEvalForm, checkpoint: e.target.value })} /></Field>
              <Field label="GPU"><input className={inputClass} value={datasetEvalForm.gpu} onChange={(e) => setDatasetEvalForm({ ...datasetEvalForm, gpu: e.target.value })} /></Field>
            </div>
            <Field label="Eval episodes"><input required className={inputClass} placeholder="0:14,29:50,115:120" value={datasetEvalForm.episode_spec} onChange={(e) => { setDatasetEvalForm({ ...datasetEvalForm, episode_spec: e.target.value }); setDatasetEvalError(null); }} /></Field>
            <div className="grid grid-cols-3 gap-4">
              <Field label="Batch size"><input type="number" min={1} className={inputClass} value={datasetEvalForm.batch_size} onChange={(e) => setDatasetEvalForm({ ...datasetEvalForm, batch_size: Number(e.target.value) })} /></Field>
              <Field label="Workers"><input type="number" min={0} className={inputClass} value={datasetEvalForm.num_workers} onChange={(e) => setDatasetEvalForm({ ...datasetEvalForm, num_workers: Number(e.target.value) })} /></Field>
              <Field label="Revision"><input className={inputClass} placeholder="main" value={datasetEvalForm.dataset_revision} onChange={(e) => setDatasetEvalForm({ ...datasetEvalForm, dataset_revision: e.target.value })} /></Field>
            </div>
            <label className="flex items-center gap-3 rounded-xl border border-white/10 bg-black/15 p-3 text-sm text-slate-300"><input type="checkbox" checked={datasetEvalForm.policy_use_bf16} onChange={(e) => setDatasetEvalForm({ ...datasetEvalForm, policy_use_bf16: e.target.checked })} className="h-4 w-4 accent-cyan-400" /> Use bfloat16</label>
            {datasetEvalError && <div className="rounded-xl border border-rose-400/20 bg-rose-400/10 p-3 text-sm text-rose-200">{datasetEvalError}</div>}
            <button className={`${buttonClass} w-full`} disabled={launchDatasetEval.isPending}>
              {launchDatasetEval.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4 fill-current" />}
              Run dataset eval
            </button>
          </form>
          <div className="mt-4"><ErrorNote error={launchDatasetEval.error || stopDatasetEval.error} /></div>
        </Panel>
        <Panel className="min-h-[420px]">
          <div className="flex items-center justify-between"><div className="flex items-center gap-2"><BarChart3 className="h-4 w-4 text-emerald-400" /><h2 className="font-bold">Dataset eval status</h2></div>{currentDatasetEval?.status === "running" && <button className={secondaryButtonClass} onClick={() => stopDatasetEval.mutate()} disabled={stopDatasetEval.isPending}><Square className="h-4 w-4" /> Stop</button>}</div>
          {!currentDatasetEval && !launchDatasetEval.isPending && <div className="grid h-[320px] place-items-center text-center text-sm text-slate-500">Choose held-out episodes to compute eval loss.</div>}
          {launchDatasetEval.isPending && <div className="grid h-[320px] place-items-center"><div className="text-center"><Loader2 className="mx-auto h-8 w-8 animate-spin text-emerald-400" /><p className="mt-3 text-sm text-slate-500">Starting dataset eval…</p></div></div>}
          {currentDatasetEval && <div className="mt-6 space-y-4"><div className={`flex items-center gap-2 rounded-xl border p-4 text-sm font-semibold ${currentDatasetEval.status === "failed" ? "border-amber-400/20 bg-amber-400/[.06] text-amber-300" : "border-emerald-400/20 bg-emerald-400/[.06] text-emerald-300"}`}>{terminal.has(currentDatasetEval.status) ? <CheckCircle2 className="h-4 w-4" /> : <Loader2 className="h-4 w-4 animate-spin" />} {currentDatasetEval.status}</div>{currentDatasetEval.metrics?.eval_loss != null && <div className="rounded-xl border border-white/10 bg-black/20 p-4"><div className="text-xs uppercase tracking-wider text-slate-500">Eval loss</div><div className="mt-1 font-mono text-3xl font-black text-white">{Number(currentDatasetEval.metrics.eval_loss).toFixed(4)}</div></div>}{currentDatasetEval.policy_ref && <div className="break-all rounded-xl bg-black/20 p-3 font-mono text-xs text-slate-400">{currentDatasetEval.policy_ref}</div>}{(currentDatasetEval.error_message || currentDatasetEval.error) && <p className="text-sm text-rose-300">{currentDatasetEval.error_message || currentDatasetEval.error}</p>}<pre className="max-h-64 overflow-auto rounded-xl bg-black/30 p-3 text-xs text-slate-400">{Array.isArray(datasetEvalLogs.data?.logs) ? datasetEvalLogs.data.logs.join("\n") : datasetEvalLogs.data?.logs || "Waiting for logs…"}</pre></div>}
        </Panel>
      </div>
    </>
  );
}

function parseEpisodeSpec(value: string): number[] {
  const trimmed = value.trim();
  if (!trimmed) throw new Error("Enter at least one episode or range.");
  const episodes: number[] = [];
  for (const rawPart of trimmed.split(",")) {
    const part = rawPart.trim();
    if (!part) continue;
    const range = part.match(/^(\d+)\s*:\s*(\d+)$/);
    if (range) {
      const start = Number(range[1]);
      const end = Number(range[2]);
      if (end <= start) throw new Error(`Episode range ${part} must end after it starts.`);
      for (let episode = start; episode < end; episode += 1) episodes.push(episode);
      continue;
    }
    if (!/^\d+$/.test(part)) throw new Error("Use episode numbers or half-open ranges like 0:14,29:50.");
    episodes.push(Number(part));
  }
  const seen = new Set<number>();
  for (const episode of episodes) {
    if (seen.has(episode)) throw new Error(`Episode ${episode} is listed more than once.`);
    seen.add(episode);
  }
  if (episodes.length === 0) throw new Error("Enter at least one episode or range.");
  return episodes;
}
