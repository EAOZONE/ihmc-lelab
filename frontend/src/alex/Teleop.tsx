import { useMutation, useQuery } from "@tanstack/react-query";
import { AlertTriangle, BarChart3, CheckCircle2, Gamepad2, Loader2, Play, Square } from "lucide-react";
import { useState } from "react";
import { alexApi, type TeleopRequest } from "./api";
import { buttonClass, ErrorNote, Field, inputClass, PageHeader, Panel, secondaryButtonClass } from "./AlexLayout";

const terminal = new Set(["completed", "failed", "stopped", "interrupted"]);

export default function Teleop() {
  const [form, setForm] = useState<TeleopRequest>({
    environment: "Isaac-Alex-Lever-Play-v0",
    teleop_device: "keyboard",
    num_envs: 1,
    sensitivity: 1,
  });
  const [teleopId, setTeleopId] = useState("");
  const launch = useMutation({
    mutationFn: (request: TeleopRequest) => alexApi.teleop(request),
    onSuccess: (result) => setTeleopId(result.id),
  });
  const result = useQuery({
    queryKey: ["teleop", teleopId], queryFn: () => alexApi.teleopStatus(teleopId), enabled: !!teleopId,
    refetchInterval: (query) => terminal.has(query.state.data?.status || "") ? false : 2000,
  });
  const logs = useQuery({
    queryKey: ["teleop-logs", teleopId], queryFn: () => alexApi.teleopLogs(teleopId), enabled: !!teleopId,
    refetchInterval: (query) => terminal.has(result.data?.status || "") ? false : 2000,
  });
  const stop = useMutation({ mutationFn: () => alexApi.stopTeleop(teleopId), onSuccess: () => result.refetch() });
  const current = result.data || launch.data;

  return (
    <>
      <PageHeader eyebrow="Direct control" title="Teleop" description="Open Isaac Lab's interactive viewer and drive Alex directly, without a trained policy in the loop." />
      <div className="grid gap-6 xl:grid-cols-[.85fr_1.15fr]">
        <Panel>
          <div className="mb-6 flex items-center gap-3"><div className="rounded-xl bg-cyan-400/10 p-3"><Gamepad2 className="text-cyan-300" /></div><div><h2 className="font-bold">New session</h2><p className="text-sm text-slate-500">Task, input device, and sensitivity</p></div></div>
          <form className="space-y-4" onSubmit={(event) => { event.preventDefault(); launch.mutate(form); }}>
            <Field label="Isaac Lab task">
              <select className={inputClass} value={form.environment} onChange={(e) => setForm({ ...form, environment: e.target.value })}>
                <option value="Isaac-Alex-Lever-Play-v0">Alex lever</option>
                <option value="Isaac-Standing-Alex-Play-v0">Standing Alex</option>
                <option value="Isaac-WalkingFlat-Alex-Play-v0">Walking flat Alex</option>
                <option value="Isaac-WalkingUneven-Alex-Play-v0">Walking uneven Alex</option>
              </select>
            </Field>
            <Field label="Input device">
              <select className={inputClass} value={form.teleop_device} onChange={(e) => setForm({ ...form, teleop_device: e.target.value as TeleopRequest["teleop_device"] })}>
                <option value="keyboard">Keyboard</option>
                <option value="spacemouse">SpaceMouse</option>
                <option value="gamepad">Gamepad</option>
                <option value="handtracking">Hand tracking (XR)</option>
              </select>
            </Field>
            <div className="grid grid-cols-2 gap-4">
              <Field label="Sensitivity"><input type="number" step="0.1" min={0.1} className={inputClass} value={form.sensitivity} onChange={(e) => setForm({ ...form, sensitivity: Number(e.target.value) })} /></Field>
              <Field label="Num envs"><input type="number" min={1} className={inputClass} value={form.num_envs} onChange={(e) => setForm({ ...form, num_envs: Number(e.target.value) })} /></Field>
            </div>
            <div className="rounded-xl border border-amber-400/20 bg-amber-400/[.07] p-4 text-sm text-amber-200">
              <div className="flex items-center gap-2 font-semibold"><AlertTriangle className="h-4 w-4" /> Known limitation</div>
              <p className="mt-2 text-xs leading-5 text-amber-100/70">The Alex lever task uses a bimanual action space that Isaac Lab's stock keyboard/SpaceMouse device doesn't drive yet. This launches the Isaac Sim viewer against the task; full input mapping is a follow-up.</p>
            </div>
            <button className={`${buttonClass} w-full`} disabled={launch.isPending}>{launch.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4 fill-current" />} Launch teleop</button>
          </form>
          <div className="mt-4"><ErrorNote error={launch.error || stop.error} /></div>
        </Panel>
        <Panel className="min-h-[500px]">
          <div className="flex items-center justify-between"><div className="flex items-center gap-2"><BarChart3 className="h-4 w-4 text-cyan-400" /><h2 className="font-bold">Session status</h2></div>{current?.status === "running" && <button className={secondaryButtonClass} onClick={() => stop.mutate()} disabled={stop.isPending}><Square className="h-4 w-4" /> Stop</button>}</div>
          {!current && !launch.isPending && <div className="grid h-[390px] place-items-center text-center text-sm text-slate-500">Choose a task and device to begin.</div>}
          {launch.isPending && <div className="grid h-[390px] place-items-center"><div className="text-center"><Loader2 className="mx-auto h-8 w-8 animate-spin text-cyan-400" /><p className="mt-3 text-sm text-slate-500">Launching Isaac Sim…</p></div></div>}
          {current && <div className="mt-6 space-y-4"><div className={`flex items-center gap-2 rounded-xl border p-4 text-sm font-semibold ${current.status === "failed" ? "border-amber-400/20 bg-amber-400/[.06] text-amber-300" : "border-emerald-400/20 bg-emerald-400/[.06] text-emerald-300"}`}>{terminal.has(current.status) ? <CheckCircle2 className="h-4 w-4" /> : <Loader2 className="h-4 w-4 animate-spin" />} {current.status}</div>{current.error_message && <p className="text-sm text-rose-300">{current.error_message}</p>}<pre className="max-h-64 overflow-auto rounded-xl bg-black/30 p-3 text-xs text-slate-400">{Array.isArray(logs.data?.logs) ? logs.data.logs.join("\n") : logs.data?.logs || "Waiting for logs…"}</pre></div>}
        </Panel>
      </div>
    </>
  );
}
