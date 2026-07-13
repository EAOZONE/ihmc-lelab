import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, CircleStop, Clock3, Cpu, Loader2, RefreshCw, TerminalSquare } from "lucide-react";
import { Link, useParams } from "react-router-dom";
import { alexApi } from "./api";
import { ErrorNote, PageHeader, Panel, secondaryButtonClass } from "./AlexLayout";

const terminalStatuses = ["completed", "failed", "stopped"];

export default function TrainingJob() {
  const { jobId = "" } = useParams();
  const queryClient = useQueryClient();
  const job = useQuery({ queryKey: ["job", jobId], queryFn: () => alexApi.job(jobId), refetchInterval: (query) => terminalStatuses.includes(query.state.data?.status || "") ? false : 2500 });
  const logs = useQuery({ queryKey: ["logs", jobId], queryFn: () => alexApi.logs(jobId), refetchInterval: (query) => terminalStatuses.includes(job.data?.status || "") ? false : 1500 });
  const stop = useMutation({ mutationFn: () => alexApi.stopJob(jobId), onSuccess: () => queryClient.invalidateQueries({ queryKey: ["job", jobId] }) });
  const logText = Array.isArray(logs.data?.logs) ? logs.data.logs.join("\n") : logs.data?.logs || "Waiting for output…";

  return (
    <>
      <Link to="/training" className="mb-5 inline-flex items-center gap-2 text-sm text-slate-400 hover:text-cyan-300"><ArrowLeft className="h-4 w-4" /> Training</Link>
      <PageHeader eyebrow={`Job ${jobId}`} title={job.data?.name || "Training run"} description="Live status and remote process output from the Alex cluster." actions={!terminalStatuses.includes(job.data?.status || "") ? <button onClick={() => stop.mutate()} disabled={stop.isPending} className={`${secondaryButtonClass} border-rose-400/20 text-rose-300 hover:bg-rose-400/10`}><CircleStop className="h-4 w-4" /> Stop job</button> : undefined} />
      <ErrorNote error={job.error || stop.error} />
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

function JobStat({ icon: Icon, label, value, spin }: { icon: typeof Cpu; label: string; value: string; spin?: boolean }) {
  return <Panel><Icon className={`mb-3 h-4 w-4 text-cyan-400 ${spin ? "animate-spin" : ""}`} /><div className="text-xs uppercase tracking-wider text-slate-500">{label}</div><div className="mt-1 truncate text-sm font-bold capitalize">{value}</div></Panel>;
}

function formatTime(value?: string | number) {
  if (!value) return "—";
  return new Date(typeof value === "number" ? value * 1000 : value).toLocaleString();
}
