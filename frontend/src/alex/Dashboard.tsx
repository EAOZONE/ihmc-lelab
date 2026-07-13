import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Activity, ArrowRight, CheckCircle2, CircleOff, Cpu, Database, Loader2, Server, Unplug } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { alexApi } from "./api";
import { buttonClass, ErrorNote, Field, inputClass, PageHeader, Panel, secondaryButtonClass } from "./AlexLayout";

export default function Dashboard() {
  const queryClient = useQueryClient();
  const setup = useQuery({ queryKey: ["setup"], queryFn: alexApi.setup, retry: false });
  const cluster = useQuery({ queryKey: ["cluster"], queryFn: alexApi.clusterStatus, refetchInterval: 5000, retry: false });
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: alexApi.jobs, refetchInterval: 5000, retry: false });
  const [form, setForm] = useState({ host: "", port: 22, username: "", password: "", expected_fingerprint: "" });
  const connect = useMutation({
    mutationFn: alexApi.connect,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["cluster"] });
      queryClient.invalidateQueries({ queryKey: ["setup"] });
    },
  });
  const disconnect = useMutation({
    mutationFn: alexApi.disconnect,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["cluster"] }),
  });
  const running = jobs.data?.filter((job) => job.status === "running").length ?? 0;

  return (
    <>
      <PageHeader eyebrow="Command center" title="Build intelligence at scale." description="Connect Alex, prepare data, and run distributed policy training from one focused workspace." />
      <div className="mb-6 grid gap-4 md:grid-cols-3">
        <Stat icon={Server} label="Cluster" value={cluster.data?.connected ? "Connected" : "Offline"} live={cluster.data?.connected} />
        <Stat icon={Cpu} label="Active jobs" value={String(running)} live={running > 0} />
        <Stat icon={CheckCircle2} label="Environment" value={setup.data?.ready ? "Ready" : "Needs setup"} live={setup.data?.ready} />
      </div>
      <div className="grid gap-6 xl:grid-cols-[1.15fr_.85fr]">
        <Panel>
          <div className="mb-5 flex items-center justify-between">
            <div>
              <h2 className="text-lg font-bold">Alex cluster</h2>
              <p className="mt-1 text-sm text-slate-500">Secure SSH connection to the training host</p>
            </div>
            <span className={`rounded-full px-3 py-1 text-xs font-bold ${cluster.data?.connected ? "bg-emerald-400/10 text-emerald-300" : "bg-slate-400/10 text-slate-400"}`}>
              {cluster.data?.connected ? cluster.data.host || "Connected" : "Disconnected"}
            </span>
          </div>
          {cluster.data?.connected ? (
            <div className="rounded-xl border border-emerald-400/15 bg-emerald-400/[.06] p-5">
              <div className="flex items-center gap-3"><CheckCircle2 className="text-emerald-400" /><div><div className="font-semibold">{cluster.data.user}@{cluster.data.host}</div><div className="text-xs text-slate-500">Telemetry refreshes automatically</div></div></div>
              <button className={`${secondaryButtonClass} mt-5`} onClick={() => disconnect.mutate()} disabled={disconnect.isPending}>
                <Unplug className="h-4 w-4" /> Disconnect
              </button>
            </div>
          ) : (
            <form className="grid gap-4 md:grid-cols-2" onSubmit={(event) => { event.preventDefault(); connect.mutate(form); }}>
              <Field label="Host"><input required className={inputClass} placeholder="alex-cluster.local" value={form.host} onChange={(e) => setForm({ ...form, host: e.target.value })} /></Field>
              <Field label="SSH port"><input required type="number" className={inputClass} value={form.port} onChange={(e) => setForm({ ...form, port: Number(e.target.value) })} /></Field>
              <Field label="Username"><input required className={inputClass} placeholder="alex" value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} /></Field>
              <Field label="SSH password"><input required type="password" className={inputClass} placeholder="••••••••" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} /></Field>
              <Field label="Host-key fingerprint" hint="Required only when this host is not already trusted in ~/.ssh/known_hosts"><input className={inputClass} placeholder="SHA256:…" value={form.expected_fingerprint} onChange={(e) => setForm({ ...form, expected_fingerprint: e.target.value })} /></Field>
              <button className={`${buttonClass} md:col-span-2`} disabled={connect.isPending}>{connect.isPending && <Loader2 className="h-4 w-4 animate-spin" />} Connect cluster</button>
            </form>
          )}
          <div className="mt-4"><ErrorNote error={connect.error || disconnect.error || cluster.error} /></div>
        </Panel>
        <Panel>
          <h2 className="text-lg font-bold">Workflow</h2>
          <p className="mt-1 text-sm text-slate-500">From raw trajectories to evaluated policy</p>
          <div className="mt-5 space-y-3">
            <WorkflowLink to="/datasets" icon={Database} title="Prepare a dataset" detail="Inspect and convert robot data" />
            <WorkflowLink to="/training" icon={Cpu} title="Launch training" detail="Any available LeRobot policy across GPUs 0–6" />
            <WorkflowLink to="/evaluation" icon={Activity} title="Evaluate checkpoint" detail="Measure policy performance" />
          </div>
        </Panel>
      </div>
      <Panel className="mt-6">
        <div className="mb-4 flex items-center justify-between"><h2 className="text-lg font-bold">Recent jobs</h2><Link className="text-sm font-semibold text-cyan-400" to="/training">View training</Link></div>
        {jobs.isLoading ? <Loader2 className="animate-spin text-cyan-400" /> : jobs.data?.length ? (
          <div className="divide-y divide-white/10">{jobs.data.slice(0, 5).map((job) => <Link to={`/training/${job.id}`} key={job.id} className="flex items-center justify-between py-3 text-sm hover:text-cyan-300"><span>{job.name || job.id}</span><span className="text-xs uppercase tracking-wider text-slate-500">{job.status}</span></Link>)}</div>
        ) : <p className="py-4 text-sm text-slate-500">No training jobs yet.</p>}
      </Panel>
    </>
  );
}

function Stat({ icon: Icon, label, value, live }: { icon: typeof Cpu; label: string; value: string; live?: boolean }) {
  return <Panel className="flex items-center gap-4"><div className="rounded-xl bg-white/5 p-3"><Icon className={live ? "text-cyan-300" : "text-slate-500"} /></div><div><div className="text-xs uppercase tracking-widest text-slate-500">{label}</div><div className="mt-1 font-bold">{value}</div></div>{live ? <span className="ml-auto h-2 w-2 rounded-full bg-emerald-400" /> : <CircleOff className="ml-auto h-4 w-4 text-slate-700" />}</Panel>;
}

function WorkflowLink({ to, icon: Icon, title, detail }: { to: string; icon: typeof Cpu; title: string; detail: string }) {
  return <Link to={to} className="group flex items-center gap-3 rounded-xl border border-white/10 bg-white/[.025] p-3 transition hover:border-cyan-300/20 hover:bg-cyan-300/[.04]"><Icon className="h-5 w-5 text-cyan-400" /><div><div className="text-sm font-semibold">{title}</div><div className="text-xs text-slate-500">{detail}</div></div><ArrowRight className="ml-auto h-4 w-4 text-slate-600 transition group-hover:translate-x-1 group-hover:text-cyan-300" /></Link>;
}
