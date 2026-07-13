import { Boxes, ChartNoAxesCombined, Database, FlaskConical, LayoutDashboard, Menu, X } from "lucide-react";
import { useState, type ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { cn } from "@/lib/utils";

const navigation = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/datasets", label: "Datasets", icon: Database },
  { to: "/training", label: "Training", icon: ChartNoAxesCombined },
  { to: "/evaluation", label: "Evaluation", icon: FlaskConical },
];

export function AlexLayout({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="min-h-screen bg-[#070b16] text-slate-100">
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(circle_at_75%_-10%,rgba(34,211,238,.12),transparent_35%),radial-gradient(circle_at_0%_80%,rgba(99,102,241,.10),transparent_30%)]" />
      <header className="sticky top-0 z-40 border-b border-white/10 bg-[#070b16]/85 backdrop-blur-xl lg:hidden">
        <div className="flex h-16 items-center justify-between px-5">
          <Brand />
          <button aria-label="Toggle navigation" onClick={() => setOpen(!open)} className="rounded-lg border border-white/10 p-2">
            {open ? <X /> : <Menu />}
          </button>
        </div>
      </header>
      <aside className={cn(
        "fixed inset-y-0 left-0 z-30 w-64 border-r border-white/10 bg-[#090e1b]/95 p-5 backdrop-blur-xl transition-transform lg:translate-x-0",
        open ? "translate-x-0 pt-20" : "-translate-x-full lg:pt-5",
      )}>
        <Brand />
        <nav className="mt-10 space-y-2">
          {navigation.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              onClick={() => setOpen(false)}
              className={({ isActive }) => cn(
                "flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition",
                isActive ? "bg-cyan-400/10 text-cyan-300 ring-1 ring-cyan-300/20" : "text-slate-400 hover:bg-white/5 hover:text-white",
              )}
            >
              <Icon className="h-4 w-4" /> {label}
            </NavLink>
          ))}
        </nav>
        <div className="absolute bottom-6 left-5 right-5 rounded-xl border border-white/10 bg-white/[.03] p-4">
          <div className="mb-2 flex items-center gap-2 text-xs font-semibold text-slate-300">
            <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_10px_#34d399]" /> ALEX CONTROL
          </div>
          <p className="text-xs leading-5 text-slate-500">Distributed robot learning workspace</p>
        </div>
      </aside>
      <main className="relative mx-auto min-h-screen max-w-[1600px] px-5 py-8 lg:ml-64 lg:px-10 lg:py-10">{children}</main>
    </div>
  );
}

function Brand() {
  return (
    <div className="flex items-center gap-3">
      <div className="grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br from-cyan-300 to-indigo-500 shadow-[0_0_30px_rgba(34,211,238,.22)]">
        <Boxes className="h-5 w-5 text-[#07101d]" />
      </div>
      <div>
        <div className="text-lg font-black tracking-[.22em]">ALEX</div>
        <div className="text-[10px] font-semibold tracking-[.28em] text-cyan-400">LAB</div>
      </div>
    </div>
  );
}

export function PageHeader({ eyebrow, title, description, actions }: {
  eyebrow: string; title: string; description: string; actions?: ReactNode;
}) {
  return (
    <div className="mb-8 flex flex-col justify-between gap-5 md:flex-row md:items-end">
      <div>
        <p className="mb-2 text-xs font-bold uppercase tracking-[.24em] text-cyan-400">{eyebrow}</p>
        <h1 className="text-3xl font-bold tracking-tight md:text-4xl">{title}</h1>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-400">{description}</p>
      </div>
      {actions}
    </div>
  );
}

export function Panel({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <section className={cn("rounded-2xl border border-white/10 bg-white/[.035] p-5 shadow-2xl shadow-black/10", className)}>{children}</section>;
}

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-2 block text-xs font-semibold uppercase tracking-wider text-slate-400">{label}</span>
      {children}
      {hint && <span className="mt-1.5 block text-xs text-slate-500">{hint}</span>}
    </label>
  );
}

export const inputClass = "h-11 w-full rounded-xl border border-white/10 bg-[#080d19] px-3 text-sm text-slate-100 outline-none transition placeholder:text-slate-600 focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-400/10";
export const buttonClass = "inline-flex h-11 items-center justify-center gap-2 rounded-xl bg-cyan-400 px-4 text-sm font-bold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-40";
export const secondaryButtonClass = "inline-flex h-11 items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 px-4 text-sm font-semibold text-slate-200 transition hover:bg-white/10 disabled:opacity-40";

export function ErrorNote({ error }: { error: unknown }) {
  if (!error) return null;
  return <div className="rounded-xl border border-rose-400/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-200">{error instanceof Error ? error.message : "Something went wrong."}</div>;
}
