import { createFileRoute } from "@tanstack/react-router";
import {
  Activity,
  Clock,
  Database,
  HardDrive,
  RotateCw,
  ScrollText,
  Server,
  TrendingUp,
  Cpu,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from "recharts";

import { AppShell } from "@/components/AppShell";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/metrics")({
  head: () => ({
    meta: [
      { title: "System Metrics — GaitVision AI" },
      {
        name: "description",
        content:
          "Operational dashboard for the gait analysis ML pipeline: inference latency, throughput, uptime, storage and service health.",
      },
      { property: "og:title", content: "System Metrics — GaitVision AI" },
      {
        property: "og:description",
        content: "Model and infrastructure health for the explainable gait analysis platform.",
      },
    ],
  }),
  component: MetricsPage,
});

const kpis = [
  {
    label: "Total Scans Processed",
    sub: "This month",
    value: "1,284",
    delta: "+12.4% vs June",
    icon: Activity,
  },
  {
    label: "Average AI Inference Time",
    sub: "Per 32-frame sequence",
    value: "4.2s",
    delta: "-0.6s vs last week",
    icon: Clock,
  },
  { label: "API Uptime", sub: "Rolling 30 days", value: "99.9%", delta: "2 incidents", icon: TrendingUp },
  {
    label: "Server Storage Used",
    sub: "RGB video archive",
    value: "45%",
    delta: "450 GB of 1 TB",
    icon: HardDrive,
    progress: 45,
  },
];

const daily = [
  { day: "Mon", videos: 38 },
  { day: "Tue", videos: 52 },
  { day: "Wed", videos: 47 },
  { day: "Thu", videos: 61 },
  { day: "Fri", videos: 73 },
  { day: "Sat", videos: 29 },
  { day: "Sun", videos: 34 },
];

const services = [
  {
    name: "Frontend Server",
    detail: "Edge runtime · eu-west-1",
    status: "Operational",
    latency: "38 ms",
    icon: Server,
    tone: "ok" as const,
  },
  {
    name: "PostgreSQL Database",
    detail: "Primary + 1 read replica",
    status: "Operational",
    latency: "12 ms",
    icon: Database,
    tone: "ok" as const,
  },
  {
    name: "FastAPI ML Inference Engine",
    detail: "GaitVision v2.4 · 2× A10G",
    status: "Degraded — high queue",
    latency: "4.2 s",
    icon: Cpu,
    tone: "warn" as const,
  },
];

function MetricsPage() {
  return (
    <AppShell>
      <div className="mx-auto max-w-[1400px] px-4 py-6 md:px-8">
        <div className="mb-6">
          <p className="text-xs font-medium tracking-wide text-primary uppercase">Operations</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight">System Metrics</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Model and infrastructure telemetry · updated 30 seconds ago
          </p>
        </div>

        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-4">
          {kpis.map((k) => (
            <div key={k.label} className="clinical-card p-5">
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-xs font-medium text-muted-foreground">{k.label}</p>
                  <p className="text-[11px] text-muted-foreground">{k.sub}</p>
                </div>
                <span className="flex size-8 items-center justify-center rounded-lg bg-primary-soft text-accent-foreground">
                  <k.icon className="size-4" />
                </span>
              </div>
              <p className="mt-4 text-3xl font-semibold tracking-tight">{k.value}</p>
              {k.progress !== undefined ? (
                <div className="mt-3">
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                    <div className="h-full rounded-full bg-primary" style={{ width: `${k.progress}%` }} />
                  </div>
                  <p className="mt-2 text-xs text-muted-foreground">{k.delta}</p>
                </div>
              ) : (
                <p className="mt-2 text-xs text-muted-foreground">{k.delta}</p>
              )}
            </div>
          ))}
        </div>

        <div className="mt-6 grid gap-6 lg:grid-cols-[1.6fr_1fr]">
          <section className="clinical-card p-5">
            <div className="flex items-end justify-between">
              <div>
                <h2 className="text-sm font-semibold">Daily Processed Videos</h2>
                <p className="mt-1 text-xs text-muted-foreground">Last 7 days · 334 clips total</p>
              </div>
              <span className="rounded-lg bg-normal px-2.5 py-1 text-xs font-medium text-normal-foreground">
                Peak Fri · 73
              </span>
            </div>
            <div className="mt-5 h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={daily} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
                  <defs>
                    <linearGradient id="videos" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="var(--color-primary)" stopOpacity={0.35} />
                      <stop offset="100%" stopColor="var(--color-primary)" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
                  <XAxis
                    dataKey="day"
                    tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }}
                    tickLine={false}
                    axisLine={{ stroke: "var(--color-border)" }}
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <RTooltip
                    contentStyle={{
                      borderRadius: 10,
                      border: "1px solid var(--color-border)",
                      fontSize: 12,
                      boxShadow: "var(--shadow-clinical-md)",
                    }}
                    formatter={(v: number) => [v, "Videos"]}
                  />
                  <Area
                    type="monotone"
                    dataKey="videos"
                    stroke="var(--color-primary)"
                    strokeWidth={2}
                    fill="url(#videos)"
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="clinical-card p-5">
            <h2 className="text-sm font-semibold">System Status</h2>
            <p className="mt-1 text-xs text-muted-foreground">Live service connectivity</p>

            <div className="mt-4 space-y-3">
              {services.map((s) => (
                <div
                  key={s.name}
                  className="flex flex-wrap items-start gap-3 rounded-xl border border-border p-3.5"
                >
                  <span className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg bg-muted text-muted-foreground">
                    <s.icon className="size-4" />
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium">{s.name}</p>
                    <p className="truncate text-xs text-muted-foreground">{s.detail}</p>
                    {s.tone !== "ok" && (
                      <div className="mt-2.5 flex flex-wrap gap-2">
                        <Button variant="outline" size="sm" className="h-7 px-2.5 text-xs">
                          <ScrollText className="size-3.5" /> View Logs
                        </Button>
                        <Button variant="outline" size="sm" className="h-7 px-2.5 text-xs">
                          <RotateCw className="size-3.5" /> Restart Instance
                        </Button>
                      </div>
                    )}
                  </div>
                  <div className="flex shrink-0 flex-col items-end gap-1">
                    <span className="flex items-center gap-1.5">
                      <span className="relative flex size-2">
                        <span
                          className={cn(
                            "absolute inline-flex size-full animate-ping rounded-full opacity-70",
                            s.tone === "ok" ? "bg-normal-foreground" : "bg-warn-foreground",
                          )}
                        />
                        <span
                          className={cn(
                            "relative inline-flex size-2 rounded-full",
                            s.tone === "ok" ? "bg-normal-foreground" : "bg-warn-foreground",
                          )}
                        />
                      </span>
                      <span
                        className={cn(
                          "text-xs font-medium",
                          s.tone === "ok" ? "text-normal-foreground" : "text-warn-foreground",
                        )}
                      >
                        {s.status}
                      </span>
                    </span>
                    <span className="text-[11px] text-muted-foreground">{s.latency}</span>
                  </div>
                </div>
              ))}
            </div>

            <p className="mt-4 text-[11px] text-muted-foreground">
              Health checks run every 15 seconds against /healthz endpoints.
            </p>
          </section>
        </div>
      </div>
    </AppShell>
  );
}
