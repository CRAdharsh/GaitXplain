import { createFileRoute } from "@tanstack/react-router";
import {
  Download,
  Info,
  Layers,
  Pause,
  Play,
  Save,
  ScanLine,
  SkipBack,
  SkipForward,
  Video,
} from "lucide-react";
import { useState } from "react";

import { AppShell } from "@/components/AppShell";
import { SkeletonHeatmap } from "@/components/SkeletonHeatmap";
import { TemporalAttentionChart, gaitPhases } from "@/components/TemporalAttentionChart";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/analysis")({
  head: () => ({
    meta: [
      { title: "Diagnostic & Explainability Output — GaitVision AI" },
      {
        name: "description",
        content:
          "Synchronized gait video, joint-level attention heatmaps, temporal attention curves and softmax abnormality classification.",
      },
      { property: "og:title", content: "Diagnostic & Explainability Output — GaitVision AI" },
      {
        property: "og:description",
        content:
          "Multi-modal explainable gait analysis: spatial attention, temporal phases and clinical classification.",
      },
    ],
  }),
  component: AnalysisPage,
});

const overlays = [
  { id: "rgb", label: "Raw RGB Video", icon: Video },
  { id: "skeleton", label: "Sparse Skeletonization", icon: ScanLine },
  { id: "attention", label: "Spatial Attention Overlay", icon: Layers },
] as const;

const classes = [
  { name: "Hemiplegic", p: 0.71, tone: "risk" },
  { name: "Spastic", p: 0.12, tone: "risk" },
  { name: "Parkinsonian", p: 0.07, tone: "warn" },
  { name: "Ataxic", p: 0.05, tone: "warn" },
  { name: "Neuropathic", p: 0.03, tone: "info" },
  { name: "Normal", p: 0.02, tone: "normal" },
];

const toneClass: Record<string, string> = {
  risk: "bg-risk text-risk-foreground",
  warn: "bg-warn text-warn-foreground",
  info: "bg-info text-info-foreground",
  normal: "bg-normal text-normal-foreground",
};

function CardTitle({ title, tip }: { title: string; tip?: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <h2 className="text-sm font-semibold">{title}</h2>
      {tip && (
        <Tooltip>
          <TooltipTrigger asChild>
            <button type="button" className="text-muted-foreground hover:text-primary">
              <Info className="size-3.5" />
            </button>
          </TooltipTrigger>
          <TooltipContent className="max-w-64">{tip}</TooltipContent>
        </Tooltip>
      )}
    </div>
  );
}

function AnalysisPage() {
  const [overlay, setOverlay] = useState<string>("attention");
  const [frame, setFrame] = useState([46]);
  const [playing, setPlaying] = useState(false);

  const current = frame[0] ?? 0;
  const phase = gaitPhases.find((p) => current >= p.start && current < p.end) ?? gaitPhases[0]!;

  return (
    <AppShell>
      <div className="mx-auto max-w-[1600px] px-4 py-6 md:px-8">
        <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="text-xs font-medium tracking-wide text-primary uppercase">Step 2 of 2</p>
            <h1 className="mt-1 text-2xl font-semibold tracking-tight">
              Diagnostic &amp; Explainability Output
            </h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Session #GS-2291 · 10-meter walk test · 120 frames @ 30 fps
            </p>
          </div>
          <span className="rounded-lg bg-normal px-3 py-1.5 text-xs font-medium text-normal-foreground">
            Analysis complete · 1.8s inference
          </span>
        </div>

        <div className="grid gap-6 xl:grid-cols-3">
          {/* Column 1 */}
          <section className="clinical-card flex flex-col p-5">
            <CardTitle
              title="Synchronized Video & Spatial Attention"
              tip="Overlays are rendered from the same downsampled T×224×224 tensor used for inference."
            />

            <div className="mt-4 flex flex-wrap gap-1.5">
              {overlays.map((o) => (
                <button
                  key={o.id}
                  onClick={() => setOverlay(o.id)}
                  className={cn(
                    "inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-all duration-200",
                    overlay === o.id
                      ? "border-primary bg-primary-soft text-accent-foreground"
                      : "border-border text-muted-foreground hover:bg-muted",
                  )}
                >
                  <o.icon className="size-3.5" />
                  {o.label}
                </button>
              ))}
            </div>

            <div className="relative mt-4 aspect-4/5 overflow-hidden rounded-xl border border-border bg-linear-to-b from-slate-100 to-slate-200">
              <div
                className={cn(
                  "absolute inset-0 transition-opacity duration-500",
                  overlay === "attention" ? "opacity-100" : "opacity-0",
                )}
                style={{
                  background:
                    "radial-gradient(circle at 63% 62%, hsl(0 82% 55% / 0.45), transparent 26%), radial-gradient(circle at 66% 78%, hsl(30 85% 55% / 0.35), transparent 22%), radial-gradient(circle at 40% 60%, hsl(220 82% 55% / 0.2), transparent 24%)",
                }}
              />
              <div
                className={cn(
                  "absolute inset-0 flex items-center justify-center transition-opacity duration-500",
                  overlay === "skeleton" ? "opacity-100" : "opacity-0",
                )}
              >
                <SkeletonHeatmap />
              </div>
              <div className="absolute top-3 left-3 rounded-md bg-surface/85 px-2 py-1 font-mono text-[11px]">
                Frame {String(current).padStart(3, "0")} / 120
              </div>
              <div className="absolute right-3 bottom-3 rounded-md bg-surface/85 px-2 py-1 font-mono text-[11px] text-muted-foreground">
                224 × 224
              </div>
            </div>

            <div className="mt-4 flex items-center gap-2">
              <Button variant="secondary" size="icon" className="size-8 rounded-full">
                <SkipBack className="size-3.5" />
              </Button>
              <Button
                size="icon"
                className="size-9 rounded-full"
                onClick={() => setPlaying((p) => !p)}
              >
                {playing ? <Pause className="size-4" /> : <Play className="size-4" />}
              </Button>
              <Button variant="secondary" size="icon" className="size-8 rounded-full">
                <SkipForward className="size-3.5" />
              </Button>
              <Slider value={frame} onValueChange={setFrame} max={100} step={1} className="flex-1" />
            </div>

            <div className="mt-3 flex items-center justify-between rounded-lg border border-border bg-muted/50 px-3 py-2">
              <span className="text-xs text-muted-foreground">Active Gait Phase</span>
              <span
                className="rounded-md px-2 py-0.5 text-xs font-medium text-foreground"
                style={{ background: phase.color }}
              >
                {phase.name}
              </span>
            </div>
          </section>

          {/* Column 2 */}
          <section className="flex flex-col gap-6">
            <div className="clinical-card p-5">
              <CardTitle
                title="Anatomic Localization Heatmap"
                tip="Per-joint gradient contribution to the predicted class. Warmer regions drove the decision more strongly."
              />
              <SkeletonHeatmap />
            </div>

            <div className="clinical-card p-5">
              <CardTitle
                title="Temporal Attention Curve"
                tip="Attention weight per frame (1…T). Peaks indicate frames most responsible for the classification."
              />
              <div className="mt-3">
                <TemporalAttentionChart />
              </div>
              <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1.5">
                {gaitPhases.map((p) => (
                  <span key={p.name} className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                    <span className="size-2.5 rounded-sm" style={{ background: p.color }} />
                    {p.name}
                  </span>
                ))}
              </div>
            </div>
          </section>

          {/* Column 3 */}
          <section className="flex flex-col gap-6">
            <div className="clinical-card p-5">
              <CardTitle
                title="Softmax Abnormality Classification"
                tip="Probabilities across the six target gait classes; values sum to 1.0."
              />

              <div className="mt-4 flex items-center justify-between rounded-xl border border-border bg-risk px-4 py-3">
                <div>
                  <p className="text-[11px] font-medium text-risk-foreground/80">Predicted class</p>
                  <p className="text-lg font-semibold text-risk-foreground">Hemiplegic Gait</p>
                </div>
                <p className="text-2xl font-semibold text-risk-foreground">71%</p>
              </div>

              <ul className="mt-4 space-y-3">
                {classes.map((c) => (
                  <li key={c.name}>
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-medium">{c.name}</span>
                      <span className="font-mono text-muted-foreground">
                        {(c.p * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="mt-1.5 h-2 w-full overflow-hidden rounded-full bg-muted">
                      <div
                        className={cn("h-full rounded-full transition-all duration-500", toneClass[c.tone])}
                        style={{
                          width: `${c.p * 100}%`,
                          backgroundColor: "currentColor",
                        }}
                      />
                    </div>
                  </li>
                ))}
              </ul>
            </div>

            <div className="clinical-card flex flex-1 flex-col p-5">
              <CardTitle title="Clinical Summary & Physician Notes" />
              <p className="mt-3 rounded-lg border border-border bg-muted/50 p-3 text-xs leading-relaxed text-foreground">
                Key findings indicate asymmetric temporal attention during the Midstance phase, localized
                predominantly in the right knee and ankle joint regions. Reduced right terminal-stance
                push-off and compensatory left-hip elevation are consistent with post-stroke hemiparetic
                gait.
              </p>

              <label className="mt-4 text-xs font-medium">Physician Remarks</label>
              <Textarea
                rows={5}
                className="mt-1.5 flex-1"
                placeholder="Add interpretation, treatment plan or follow-up interval…"
                defaultValue="Recommend AFO trial and 6-week gait retraining protocol. Re-scan at follow-up."
              />

              <div className="mt-4 flex flex-wrap gap-2">
                <Button className="flex-1">
                  <Save className="size-4" /> Save to Patient Record
                </Button>
                <Button variant="outline" className="flex-1">
                  <Download className="size-4" /> Download PDF Report
                </Button>
              </div>
            </div>
          </section>
        </div>
      </div>
    </AppShell>
  );
}
