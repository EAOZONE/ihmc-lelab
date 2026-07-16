import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  Cpu,
  Database,
  Loader2,
  LockKeyhole,
  Play,
  RefreshCw,
  Server,
  Thermometer,
  Zap,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  alexApi,
  type GpuTelemetry,
  type LeRobotTrainingConfig,
  type PolicyCapability,
  type TrainingRequest,
} from "./api";
import { buttonClass, ErrorNote, Field, inputClass, PageHeader, Panel } from "./AlexLayout";

const blankGpu = (index: number): GpuTelemetry => ({
  index,
  name: "Awaiting telemetry",
  utilization: 0,
  memory_used_mb: 0,
  memory_total_mb: 0,
  temperature_c: 0,
  power_w: 0,
  processes: [],
});

const initialConfig: LeRobotTrainingConfig = {
  kind: "lerobot",
  dataset_repo_id: "",
  model_repo_id: "",
  policy_type: "act",
  steps: 10000,
  batch_size: 8,
  seed: 1000,
  num_workers: 4,
  log_freq: 250,
  save_freq: 1000,
  save_checkpoint: true,
  policy_use_amp: false,
  wandb_enable: false,
  dataset_image_transforms_enable: false,
};

export default function Training() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const cluster = useQuery({ queryKey: ["cluster"], queryFn: alexApi.clusterStatus, refetchInterval: 5000, retry: false });
  const gpuQuery = useQuery({ queryKey: ["gpus"], queryFn: alexApi.gpus, enabled: !!cluster.data?.connected, refetchInterval: 2500, retry: false });
  const datasets = useQuery({ queryKey: ["alex-datasets"], queryFn: alexApi.datasets, retry: false });
  const [ssh, setSsh] = useState({ host: "", port: 22, username: "", password: "", expected_fingerprint: "" });
  const [selected, setSelected] = useState<number[]>([]);
  const [name, setName] = useState("alex-policy");
  const [config, setConfig] = useState<LeRobotTrainingConfig>(initialConfig);
  const [episodeSpec, setEpisodeSpec] = useState("");
  const [episodeError, setEpisodeError] = useState<string | null>(null);
  const datasetRepoValid = /^[^\s/]+\/[^\s/]+$/.test(config.dataset_repo_id);
  const capabilities = useQuery({
    queryKey: ["training-capabilities", config.dataset_repo_id],
    queryFn: () => alexApi.trainingCapabilities(config.dataset_repo_id),
    enabled: !!cluster.data?.connected && datasetRepoValid,
    retry: false,
  });
  const connect = useMutation({ mutationFn: alexApi.connect, onSuccess: () => queryClient.invalidateQueries({ queryKey: ["cluster"] }) });
  const launch = useMutation({ mutationFn: alexApi.train, onSuccess: (job) => navigate(`/training/${job.id}`) });
  const gpus = useMemo(
    () => Array.from({ length: 7 }, (_, index) => gpuQuery.data?.find((gpu) => gpu.index === index) || blankGpu(index)),
    [gpuQuery.data],
  );
  const selectedPolicy = capabilities.data?.policies.find((policy) => policy.type === config.policy_type);

  useEffect(() => {
    if (!capabilities.data?.policies.length) return;
    const current = capabilities.data.policies.find((policy) => policy.type === config.policy_type);
    if (current?.available && current.compatible) return;
    const first = capabilities.data.policies.find((policy) => policy.available && policy.compatible);
    if (first) setConfig((value) => applyPolicyDefaults(value, first));
  }, [capabilities.data, config.policy_type]);

  const update = <K extends keyof LeRobotTrainingConfig>(key: K, value: LeRobotTrainingConfig[K]) => {
    setConfig((current) => ({ ...current, [key]: value }));
  };
  const chooseDataset = (repoId: string) => {
    setConfig((current) => ({
      ...current,
      dataset_repo_id: repoId,
      model_repo_id: repoId ? `${repoId}-${current.policy_type}` : "",
    }));
  };
  const choosePolicy = (policy: PolicyCapability) => {
    setConfig((current) => {
      const next = applyPolicyDefaults(current, policy);
      return {
        ...next,
        model_repo_id: current.dataset_repo_id ? `${current.dataset_repo_id}-${policy.type}` : current.model_repo_id,
      };
    });
  };
  const submit = () => {
    let datasetEpisodes: number[] | undefined;
    try {
      datasetEpisodes = parseEpisodeSpec(episodeSpec);
      setEpisodeError(null);
    } catch (error) {
      setEpisodeError(error instanceof Error ? error.message : "Invalid episode selection");
      return;
    }
    const payloadConfig: LeRobotTrainingConfig = {
      ...config,
      dataset_episodes: datasetEpisodes,
    };
    const payload: TrainingRequest = { name, config: payloadConfig, gpus: selected.map(String) };
    launch.mutate(payload);
  };
  const policyBlocked = !selectedPolicy?.available || !selectedPolicy.compatible;
  const canLaunch = !!cluster.data?.connected && selected.length > 0 && datasetRepoValid && !!config.model_repo_id && !!selectedPolicy && !policyBlocked;

  return (
    <>
      <PageHeader
        eyebrow="Distributed compute"
        title="Training"
        description="Train any policy available in the pinned LeRobot image on your Alex datasets and selected gpu2 accelerators."
        actions={<div className="flex items-center gap-2 text-xs text-slate-500"><RefreshCw className={`h-3.5 w-3.5 ${gpuQuery.isFetching ? "animate-spin text-cyan-400" : ""}`} /> 2.5s live refresh</div>}
      />
      {!cluster.data?.connected && <Panel className="mb-6 border-cyan-400/20">
        <div className="mb-4 flex items-center gap-2"><LockKeyhole className="h-4 w-4 text-cyan-300" /><h2 className="font-bold">Connect to Alex via SSH</h2></div>
        <form className="grid gap-3 md:grid-cols-5" onSubmit={(event) => { event.preventDefault(); connect.mutate(ssh); }}>
          <input required aria-label="Host" className={inputClass} placeholder="Host" value={ssh.host} onChange={(event) => setSsh({ ...ssh, host: event.target.value })} />
          <input required aria-label="Username" className={inputClass} placeholder="Username" value={ssh.username} onChange={(event) => setSsh({ ...ssh, username: event.target.value })} />
          <input required aria-label="Password" type="password" className={inputClass} placeholder="SSH password" value={ssh.password} onChange={(event) => setSsh({ ...ssh, password: event.target.value })} />
          <input required aria-label="Port" type="number" className={inputClass} value={ssh.port} onChange={(event) => setSsh({ ...ssh, port: Number(event.target.value) })} />
          <input aria-label="Host fingerprint" className={inputClass} placeholder="SHA256:…" value={ssh.expected_fingerprint} onChange={(event) => setSsh({ ...ssh, expected_fingerprint: event.target.value })} />
          <button className={`${buttonClass} md:col-span-5`} disabled={connect.isPending}>{connect.isPending && <Loader2 className="h-4 w-4 animate-spin" />} Connect</button>
        </form>
        <div className="mt-3"><ErrorNote error={connect.error || cluster.error} /></div>
      </Panel>}

      <Panel className="mb-6">
        <div className="mb-4 flex items-center justify-between"><div><h2 className="font-bold">GPU fleet</h2><p className="mt-1 text-xs text-slate-500">Select one or more available accelerators</p></div><span className="text-xs text-slate-500">{selected.length} selected</span></div>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">{gpus.map((gpu) => <GpuCard key={gpu.index} gpu={gpu} selected={selected.includes(gpu.index)} disabled={!cluster.data?.connected} onToggle={() => setSelected((current) => current.includes(gpu.index) ? current.filter((id) => id !== gpu.index) : [...current, gpu.index])} />)}</div>
        <div className="mt-4"><ErrorNote error={gpuQuery.error} /></div>
      </Panel>

      <div className="grid gap-6 xl:grid-cols-[.75fr_1.25fr]">
        <Panel>
          <div className="flex items-center gap-2"><Database className="h-4 w-4 text-cyan-400" /><h2 className="font-bold">Dataset and policy</h2></div>
          <div className="mt-5 space-y-4">
            <Field label="Alex dataset on Hugging Face">
              <input
                required
                list="alex-hub-datasets"
                className={inputClass}
                placeholder={datasets.isLoading ? "Loading datasets…" : "owner/alex-dataset"}
                value={config.dataset_repo_id}
                onChange={(event) => chooseDataset(event.target.value)}
              />
              <datalist id="alex-hub-datasets">
                {datasets.data?.map((dataset) => <option key={dataset.repo_id} value={dataset.repo_id}>{dataset.private ? "private" : "public"}</option>)}
              </datalist>
            </Field>
            <Field label="Policy">
              <select
                className={inputClass}
                value={config.policy_type}
                disabled={!config.dataset_repo_id || capabilities.isLoading}
                onChange={(event) => {
                  const policy = capabilities.data?.policies.find((item) => item.type === event.target.value);
                  if (policy) choosePolicy(policy);
                }}
              >
                {!capabilities.data && <option value="act">{capabilities.isLoading ? "Detecting policies…" : "Select a dataset first"}</option>}
                {capabilities.data?.policies.map((policy) => <option key={policy.type} value={policy.type} disabled={!policy.available || !policy.compatible}>{policy.label}{!policy.available || !policy.compatible ? " · unavailable" : ""}</option>)}
              </select>
            </Field>
            {selectedPolicy && (!selectedPolicy.available || !selectedPolicy.compatible) && <div className="rounded-xl border border-amber-400/20 bg-amber-400/10 p-3 text-sm text-amber-200">{selectedPolicy.unavailable_reason || selectedPolicy.compatibility_reason}</div>}
            {capabilities.data?.dataset_warnings?.map((warning) => <div key={warning} className="rounded-xl border border-amber-400/20 bg-amber-400/10 p-3 text-sm text-amber-200"><AlertTriangle className="mr-2 inline h-4 w-4" />{warning}</div>)}
            {capabilities.data && <div className="rounded-xl border border-white/10 bg-black/20 p-4 text-xs leading-5 text-slate-500"><Server className="mb-2 h-4 w-4 text-cyan-400" />LeRobot {capabilities.data.lerobot_version} · PyTorch {capabilities.data.torch_version || "unknown"} · CUDA {capabilities.data.torch_cuda_version || "unknown"}<br />{capabilities.data.image}</div>}
            <ErrorNote error={datasets.error || capabilities.error} />
          </div>
        </Panel>

        <Panel>
          <h2 className="font-bold">Run configuration</h2>
          <div className="mt-5 grid gap-4 md:grid-cols-2">
            <Field label="Run name"><input className={inputClass} value={name} onChange={(event) => setName(event.target.value)} /></Field>
            <Field label="Hugging Face model repo"><input required className={inputClass} placeholder="owner/alex-act" value={config.model_repo_id} onChange={(event) => update("model_repo_id", event.target.value)} /></Field>
            <Field label="Pretrained policy (optional)"><input className={inputClass} placeholder="owner/model or local image path" value={config.policy_pretrained_path || ""} onChange={(event) => update("policy_pretrained_path", event.target.value || undefined)} /></Field>
            <Field label="Training episodes (optional)">
              <input
                className={inputClass}
                placeholder="14:29,50:115"
                value={episodeSpec}
                onChange={(event) => {
                  setEpisodeSpec(event.target.value);
                  setEpisodeError(null);
                }}
              />
            </Field>
            <NumberField label="Training steps" value={config.steps} min={1} onChange={(value) => update("steps", value)} />
            <NumberField label="Batch size per GPU" value={config.batch_size} min={1} onChange={(value) => update("batch_size", value)} />
            <NumberField label="Data-loader workers" value={config.num_workers} min={0} onChange={(value) => update("num_workers", value)} />
            <NumberField label="Save every steps" value={config.save_freq} min={1} onChange={(value) => update("save_freq", value)} />
            <NumberField label="Log every steps" value={config.log_freq} min={1} onChange={(value) => update("log_freq", value)} />
            {config.policy_type === "groot" && <>
              <Field label="GR00T N1.7 base model"><input className={inputClass} value={config.policy_base_model_path || ""} onChange={(event) => update("policy_base_model_path", event.target.value || undefined)} /></Field>
              <Field label="Embodiment"><input className={inputClass} value={config.policy_embodiment_tag || ""} onChange={(event) => update("policy_embodiment_tag", event.target.value || undefined)} /></Field>
              <NumberField label="Chunk size" value={config.policy_chunk_size || 16} min={1} onChange={(value) => update("policy_chunk_size", value)} />
              <NumberField label="Action steps" value={config.policy_n_action_steps || 16} min={1} onChange={(value) => update("policy_n_action_steps", value)} />
            </>}
          </div>
          <div className="mt-5 grid gap-3 md:grid-cols-2">
            <Check label="Enable image augmentation" checked={config.dataset_image_transforms_enable} onChange={(value) => update("dataset_image_transforms_enable", value)} />
            <Check label="Enable Weights & Biases" checked={config.wandb_enable} onChange={(value) => update("wandb_enable", value)} />
            {config.policy_type === "groot" && <>
              <Check label="Use relative actions" checked={config.policy_use_relative_actions ?? false} onChange={(value) => update("policy_use_relative_actions", value)} />
              <Check label="Use bfloat16" checked={config.policy_use_bf16 ?? true} onChange={(value) => update("policy_use_bf16", value)} />
            </>}
          </div>
          {config.wandb_enable && <div className="mt-4"><Field label="W&B project"><input className={inputClass} value={config.wandb_project || ""} onChange={(event) => update("wandb_project", event.target.value || undefined)} /></Field></div>}
          {selected.some((index) => gpus[index]?.occupied) && <div className="mt-4 flex gap-2 rounded-xl border border-amber-400/20 bg-amber-400/10 p-3 text-sm text-amber-200"><AlertTriangle className="h-4 w-4 shrink-0" /> Selected GPUs have active processes. Launching may cause contention.</div>}
          {episodeError && <div className="mt-4 rounded-xl border border-rose-400/20 bg-rose-400/10 p-3 text-sm text-rose-200">{episodeError}</div>}
          <div className="mt-5"><ErrorNote error={launch.error} /></div>
          <button className={`${buttonClass} mt-5 w-full`} disabled={!canLaunch || launch.isPending} onClick={submit}>{launch.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4 fill-current" />} Launch on {selected.length} GPU{selected.length === 1 ? "" : "s"}</button>
        </Panel>
      </div>
    </>
  );
}

function parseEpisodeSpec(value: string): number[] | undefined {
  const trimmed = value.trim();
  if (!trimmed) return undefined;

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
    if (!/^\d+$/.test(part)) {
      throw new Error("Use episode numbers or half-open ranges like 14:29,50:115.");
    }
    episodes.push(Number(part));
  }

  if (episodes.length === 0) return undefined;
  const seen = new Set<number>();
  for (const episode of episodes) {
    if (seen.has(episode)) throw new Error(`Episode ${episode} is listed more than once.`);
    seen.add(episode);
  }
  return episodes;
}

function applyPolicyDefaults(config: LeRobotTrainingConfig, policy: PolicyCapability): LeRobotTrainingConfig {
  const next: LeRobotTrainingConfig = {
    ...config,
    policy_type: policy.type,
    policy_pretrained_path: undefined,
    policy_base_model_path: undefined,
    policy_embodiment_tag: undefined,
    policy_chunk_size: undefined,
    policy_n_action_steps: undefined,
    policy_use_relative_actions: undefined,
    policy_relative_exclude_joints: undefined,
    policy_use_bf16: undefined,
  };
  for (const field of policy.fields) {
    if (field.default !== null && field.default !== undefined) {
      (next as unknown as Record<string, unknown>)[field.name] = field.default;
    }
  }
  if (policy.type === "groot") {
    next.policy_relative_exclude_joints = ["gripper"];
    next.dataset_image_transforms_enable = true;
  }
  return next;
}

function NumberField({ label, value, min, onChange }: { label: string; value: number; min: number; onChange: (value: number) => void }) {
  return <Field label={label}><input type="number" min={min} className={inputClass} value={value} onChange={(event) => onChange(Number(event.target.value))} /></Field>;
}

function Check({ label, checked, onChange }: { label: string; checked: boolean; onChange: (value: boolean) => void }) {
  return <label className="flex items-center gap-3 rounded-xl border border-white/10 bg-black/15 p-3 text-sm text-slate-300"><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} className="h-4 w-4 accent-cyan-400" />{label}</label>;
}

function GpuCard({ gpu, selected, disabled, onToggle }: { gpu: GpuTelemetry; selected: boolean; disabled: boolean; onToggle: () => void }) {
  const occupied = gpu.occupied ?? gpu.processes.length > 0;
  const memory = gpu.memory_total_mb ? Math.round(gpu.memory_used_mb / gpu.memory_total_mb * 100) : 0;
  const usedVram = (gpu.memory_used_mb / 1024).toFixed(1);
  const totalVram = gpu.memory_total_mb ? (gpu.memory_total_mb / 1024).toFixed(1) : "—";
  return <button disabled={disabled} onClick={onToggle} className={`rounded-xl border p-4 text-left transition ${selected ? "border-cyan-300/60 bg-cyan-300/[.08] ring-1 ring-cyan-300/20" : "border-white/10 bg-black/15 hover:border-white/20"} disabled:cursor-not-allowed disabled:opacity-45`}>
    <div className="flex justify-between"><span className="text-xs font-black tracking-wider text-cyan-300">GPU {gpu.index}</span>{occupied && <span className="flex items-center gap-1 text-[10px] font-bold text-amber-300"><AlertTriangle className="h-3 w-3" /> OCCUPIED</span>}</div>
    <div className="mt-1 truncate text-xs text-slate-500">{gpu.name}</div>
    <div className="mt-4 text-2xl font-black text-white">{usedVram} <span className="text-sm font-medium text-slate-500">/ {totalVram} GiB VRAM</span></div>
    <div className="mt-4 space-y-3"><Metric label="GPU utilization" value={`${gpu.utilization}%`} amount={gpu.utilization} /><Metric label="VRAM used" value={`${memory}%`} amount={memory} /></div>
    <div className="mt-4 flex gap-4 border-t border-white/10 pt-3 text-[11px] text-slate-400"><span className="flex items-center gap-1"><Thermometer className="h-3 w-3" />{gpu.temperature_c}°C</span><span className="flex items-center gap-1"><Zap className="h-3 w-3" />{gpu.power_w}W</span><span className="ml-auto flex items-center gap-1"><Cpu className="h-3 w-3" />{gpu.processes.length}</span></div>
  </button>;
}

function Metric({ label, value, amount }: { label: string; value: string; amount: number }) {
  return <div><div className="mb-1 flex justify-between text-[10px] text-slate-500"><span>{label}</span><span>{value}</span></div><div className="h-1 overflow-hidden rounded-full bg-white/10"><div className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-indigo-400" style={{ width: `${Math.min(100, amount)}%` }} /></div></div>;
}
