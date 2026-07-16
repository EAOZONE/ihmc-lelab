import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Activity, ArrowLeft, CircleStop, Clock3, Cpu, Loader2, RefreshCw, TerminalSquare, TrendingDown } from "lucide-react";
import { Link, useParams } from "react-router-dom";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { alexApi, type TrainingMetricPoint } from "./api";
import { ErrorNote, PageHeader, Panel, secondaryButtonClass } from "./AlexLayout";

const terminalStatuses = ["completed", "failed", "stopped"];

export default function TrainingJob() {
  const { jobId = "" } = useParams();
  const queryClient = useQueryClient();
  const job = useQuery({ queryKey: ["job", jobId], queryFn: () => alexApi.job(jobId), refetchInterval: (query) => terminalStatuses.includes(query.state.data?.status || "") ? false : 2500 });
  const logs = useQuery({ queryKey: ["logs", jobId], queryFn: () => alexApi.logs(jobId), refetchInterval: (query) => terminalStatuses.includes(job.data?.status || "") ? false : 1500 });
  const metrics = useQuery({ queryKey: ["metrics-history", jobId], queryFn: () => alexApi.metricsHistory(jobId), refetchInterval: () => terminalStatuses.includes(job.data?.status || "") ? false : 2500 });
  const stop = useMutation({ mutationFn: () => alexApi.stopJob(jobId), onSuccess: () => queryClient.invalidateQueries({ queryKey: ["job", jobId] }) });
  const logText = Array.isArray(logs.data?.logs) ? logs.data.logs.join("\n") : logs.data?.logs || "Waiting for output…";

  return (
    <>
      <Link to="/training" className="mb-5 inline-flex items-center gap-2 text-sm text-slate-400 hover:text-cyan-300"><ArrowLeft className="h-4 w-4" /> Training</Link>
      <PageHeader eyebrow={`Job ${jobId}`} title={job.data?.name || "Training run"} description="Live status and remote process output from the Alex cluster." actions={!terminalStatuses.includes(job.data?.status || "") ? <button onClick={() => stop.mutate()} disabled={stop.isPending} className={`${secondaryButtonClass} border-rose-400/20 text-rose-300 hover:bg-rose-400/10`}><CircleStop className="h-4 w-4" /> Stop job</button> : undefined} />
      <ErrorNote error={job.error || stop.error} />
      <TrainingCharts points={metrics.data?.points || []} loading={metrics.isLoading} />
      <div className="mt-4"><ErrorNote error={metrics.error} /></div>
      <div className="mt-6 grid gap-4 md:grid-cols-4">
        <JobStat icon={RefreshCw} label="Status" value={job.data?.status || "Loading"} spin={job.data?.status === "running"} />
        <JobStat icon={Cpu} label="GPUs" value={job.data?.gpus?.join(", ") || "—"} />
        <JobStat icon={Clock3} label="Started" value={formatTime(job.data?.started_at)} />
        <JobStat icon={Clock3} label="Finished" value={formatTime(job.data?.finished_at)} />
      </div>
      {typeof job.data?.progress === "number" && <Panel className="mt-6"><div className="mb-2 flex justify-between text-xs text-slate-400"><span>Progress</span><span>{Math.round(job.data.progress)}%</span></div><div className="h-2 overflow-hidden rounded-full bg-white/10"><div className="h-full bg-gradient-to-r from-cyan-400 to-indigo-500" style={{ width: `${job.data.progress}%` }} /></div></Panel>}
      <Panel className="mt-6 p-0">
        <div className="flex items-center justify-between border-b border-white/10 px-5 py-4"><div className="flex items-center gap-2 text-sm font-semibold"><TerminalSquare className="h-4 w-4 text-cyan-400" /> Live logs</div>{logs.isFetching && <Loader2 className="h-4 w-4 animate-spin text-cyan-400" />}</div>
        <pre className="max-h-[560px] min-h-[360px] overflow-auto p-5 font-mono text-xs leading-6 text-slate-300">{logText}</pre>
      </Panel>
      <div className="mt-4"><ErrorNote error={logs.error} /></div>
    </>
  );
}

function TrainingCharts({ points, loading }: { points: TrainingMetricPoint[]; loading: boolean }) {
  const loss = points.filter((point) => point.loss != null);
  const learningRate = points.filter((point) => point.lr != null);
  return (
    <div className="mt-6 grid gap-4 lg:grid-cols-2">
      <MetricChart
        title="Loss"
        icon={TrendingDown}
        data={loss}
        dataKey="loss"
        color="#34d399"
        current={loss.at(-1)?.loss}
        loading={loading}
        format={(value) => value.toFixed(4)}
      />
      <MetricChart
        title="Learning rate"
        icon={Activity}
        data={learningRate}
        dataKey="lr"
        color="#fb923c"
        current={learningRate.at(-1)?.lr}
        loading={loading}
        format={(value) => value.toExponential(2)}
      />
    </div>
  );
}

function MetricChart({ title, icon: Icon, data, dataKey, color, current, loading, format }: {
  title: string;
  icon: typeof Activity;
  data: TrainingMetricPoint[];
  dataKey: "loss" | "lr";
  color: string;
  current: number | null | undefined;
  loading: boolean;
  format: (value: number) => string;
}) {
  return (
    <Panel>
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-semibold"><Icon className="h-4 w-4" style={{ color }} /> {title}</div>
        <span className="font-mono text-xs text-slate-400">{current == null ? "—" : format(current)}</span>
      </div>
      <div className="h-52">
        {data.length === 0 ? (
          <div className="grid h-full place-items-center text-sm text-slate-500">{loading ? "Loading metrics…" : "Waiting for the first metric tick…"}</div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 8, right: 12, left: 4, bottom: 0 }}>
              <XAxis dataKey="step" stroke="#475569" tick={{ fill: "#94a3b8", fontSize: 11 }} tickLine={false} />
              <YAxis stroke="#475569" tick={{ fill: "#94a3b8", fontSize: 11 }} tickLine={false} width={58} tickFormatter={dataKey === "lr" ? (value: number) => value.toExponential(0) : undefined} />
              <Tooltip
                contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 10 }}
                labelStyle={{ color: "#cbd5e1" }}
                formatter={(value: number) => [format(value), title]}
                labelFormatter={(step) => `Step ${Number(step).toLocaleString()}`}
              />
              <Line type="monotone" dataKey={dataKey} stroke={color} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </Panel>
  );
}

function JobStat({ icon: Icon, label, value, spin }: { icon: typeof Cpu; label: string; value: string; spin?: boolean }) {
  return <Panel><Icon className={`mb-3 h-4 w-4 text-cyan-400 ${spin ? "animate-spin" : ""}`} /><div className="text-xs uppercase tracking-wider text-slate-500">{label}</div><div className="mt-1 truncate text-sm font-bold capitalize">{value}</div></Panel>;
}

function formatTime(value?: string | number) {
  if (!value) return "—";
  return new Date(typeof value === "number" ? value * 1000 : value).toLocaleString();
}
