const API_BASE = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

export class AlexApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
    this.name = "AlexApiError";
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init.headers },
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = body.detail || body.message || message;
    } catch {
      // Keep the HTTP status when the backend has no JSON error body.
    }
    throw new AlexApiError(message, response.status);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export type JobStatus = "queued" | "running" | "completed" | "failed" | "stopped";

export interface SetupStatus {
  ready: boolean;
  message?: string;
  dependencies?: Record<string, boolean>;
}

export interface ClusterStatus {
  connected: boolean;
  host?: string;
  user?: string;
  message?: string;
}

export interface GpuProcess {
  pid: number;
  name: string;
  memory_mb: number;
  user?: string;
}

export interface GpuTelemetry {
  index: number;
  name: string;
  uuid?: string;
  utilization: number;
  memory_used_mb: number;
  memory_total_mb: number;
  temperature_c: number;
  power_w: number;
  power_limit_w?: number;
  occupied?: boolean;
  processes: GpuProcess[];
}

export interface DatasetInspection {
  path: string;
  valid: boolean;
  format?: string;
  episodes?: number;
  frames?: number;
  size_bytes?: number;
  features?: string[];
  warnings?: string[];
  message?: string;
}

export interface HubDataset {
  repo_id: string;
  last_modified?: string;
  private: boolean;
  source: "hub" | "both";
}

export interface DatasetConversion {
  id?: string;
  source: string;
  destination: string;
  status: string;
  message?: string;
}

export interface AlexJob {
  id: string;
  name?: string;
  method?: "groot" | "ccil" | string;
  status: JobStatus;
  created_at?: string;
  started_at?: string | number;
  finished_at?: string | number;
  progress?: number;
  metrics?: TrainingMetrics;
  gpus?: number[];
  config?: Record<string, unknown>;
  error?: string;
}

export interface TrainingMetrics {
  current_step: number;
  total_steps: number;
  current_loss: number | null;
  current_lr: number | null;
  grad_norm: number | null;
  eta_seconds: number | null;
}

export interface TrainingMetricPoint {
  step: number;
  loss: number | null;
  lr: number | null;
  grad_norm: number | null;
}

export interface JobLogs {
  logs: string[] | string;
  cursor?: string;
}

export interface PolicyField {
  name: string;
  label: string;
  type: "string" | "integer" | "boolean";
  default?: string | number | boolean | null;
}

export interface PolicyCapability {
  type: string;
  label: string;
  available: boolean;
  unavailable_reason?: string;
  compatible: boolean;
  compatibility_reason?: string;
  fields: PolicyField[];
}

export interface TrainingCapabilities {
  image: string;
  lerobot_version: string;
  torch_version?: string;
  torch_cuda_version?: string;
  cuda_device_count?: number;
  dataset_repo_id?: string;
  dataset_warnings?: string[];
  groot_relative_actions_ready?: boolean;
  groot_relative_actions_reason?: string;
  policies: PolicyCapability[];
}

export interface LeRobotTrainingConfig {
  kind: "lerobot";
  dataset_repo_id: string;
  model_repo_id: string;
  policy_type: string;
  policy_pretrained_path?: string;
  steps: number;
  batch_size: number;
  seed?: number;
  num_workers: number;
  log_freq: number;
  save_freq: number;
  save_checkpoint: boolean;
  policy_use_amp: boolean;
  wandb_enable: boolean;
  wandb_project?: string;
  dataset_image_transforms_enable: boolean;
  policy_base_model_path?: string;
  policy_embodiment_tag?: string;
  policy_chunk_size?: number;
  policy_n_action_steps?: number;
  policy_use_relative_actions?: boolean;
  policy_relative_exclude_joints?: string[];
  policy_use_bf16?: boolean;
}

export interface TrainingRequest {
  name: string;
  gpus: string[];
  config: LeRobotTrainingConfig;
}

export interface RolloutRequest {
  target: "sim" | "arena" | "robot";
  inference_location?: "remote" | "local";
  job_id?: string;
  policy_ref?: string;
  checkpoint: string;
  dataset_repo_id?: string;
  gpu: string;
  task: string;
  environment: string;
  embodiment: string;
  usd?: string;
  num_episodes: number;
  video: boolean;
  camera_video: boolean;
}

export interface RolloutResult {
  id: string;
  status: string;
  metrics?: Record<string, number | string>;
  artifacts?: string[];
  error_message?: string;
  blockers?: string[];
  policy_ref?: string;
}

export interface TeleopRequest {
  environment: string;
  teleop_device: "keyboard" | "spacemouse" | "gamepad" | "handtracking";
  num_envs: number;
  sensitivity: number;
}

export interface TeleopResult {
  id: string;
  status: string;
  pid?: number;
  error_message?: string;
}

export const alexApi = {
  setup: () => request<SetupStatus>("/alex/setup"),
  connect: (body: { host: string; port: number; username: string; password: string; expected_fingerprint?: string }) =>
    request<ClusterStatus>("/alex/cluster/connect", { method: "POST", body: JSON.stringify(body) }),
  disconnect: () => request<ClusterStatus>("/alex/cluster/disconnect", { method: "POST" }),
  clusterStatus: () => request<ClusterStatus>("/alex/cluster/status"),
  gpus: () => request<GpuTelemetry[]>("/alex/cluster/gpus"),
  datasets: () => request<HubDataset[]>("/alex/datasets"),
  trainingCapabilities: (datasetRepoId?: string) =>
    request<TrainingCapabilities>(
      `/alex/training/capabilities${datasetRepoId ? `?dataset_repo_id=${encodeURIComponent(datasetRepoId)}` : ""}`,
    ),
  inspectDataset: (body: { path: string }) =>
    request<DatasetInspection>("/alex/datasets/inspect", { method: "POST", body: JSON.stringify(body) }),
  convertDataset: (body: { source: string; destination: string; format: string }) =>
    request<DatasetConversion>("/alex/datasets/convert", { method: "POST", body: JSON.stringify(body) }),
  jobs: () => request<AlexJob[]>("/alex/jobs"),
  job: (id: string) => request<AlexJob>(`/alex/jobs/${encodeURIComponent(id)}`),
  logs: (id: string) => request<JobLogs>(`/alex/jobs/${encodeURIComponent(id)}/logs`),
  metricsHistory: (id: string) =>
    request<{ points: TrainingMetricPoint[] }>(`/alex/jobs/${encodeURIComponent(id)}/metrics-history`),
  stopJob: (id: string) =>
    request<AlexJob>(`/alex/jobs/${encodeURIComponent(id)}/stop`, { method: "POST" }),
  train: (body: TrainingRequest) =>
    request<AlexJob>("/alex/training", { method: "POST", body: JSON.stringify(body) }),
  rollout: (body: RolloutRequest) =>
    request<RolloutResult>("/alex/rollouts", { method: "POST", body: JSON.stringify(body) }),
  rolloutStatus: (id: string) => request<RolloutResult>(`/alex/rollouts/${encodeURIComponent(id)}`),
  rolloutLogs: (id: string) => request<JobLogs>(`/alex/rollouts/${encodeURIComponent(id)}/logs`),
  stopRollout: (id: string) =>
    request<RolloutResult>(`/alex/rollouts/${encodeURIComponent(id)}/stop`, { method: "POST" }),
  teleop: (body: TeleopRequest) =>
    request<TeleopResult>("/alex/teleop", { method: "POST", body: JSON.stringify(body) }),
  teleopStatus: (id: string) => request<TeleopResult>(`/alex/teleop/${encodeURIComponent(id)}`),
  teleopLogs: (id: string) => request<JobLogs>(`/alex/teleop/${encodeURIComponent(id)}/logs`),
  stopTeleop: (id: string) =>
    request<TeleopResult>(`/alex/teleop/${encodeURIComponent(id)}/stop`, { method: "POST" }),
};
