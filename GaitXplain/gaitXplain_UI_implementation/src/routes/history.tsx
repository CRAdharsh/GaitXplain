import { createFileRoute } from "@tanstack/react-router";
import { ChevronRight, FileText, Folder, FolderOpen, ArrowUpRight, Waypoints } from "lucide-react";
import { useState } from "react";

import { AppShell } from "@/components/AppShell";
import { TemporalAttentionChart } from "@/components/TemporalAttentionChart";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/history")({
  head: () => ({
    meta: [
      { title: "Analytical History Tree — GaitVision AI" },
      {
        name: "description",
        content:
          "Browse historical gait analyses in a hierarchical tree by year, quarter and patient with quick explainability previews.",
      },
      { property: "og:title", content: "Analytical History Tree — GaitVision AI" },
      {
        property: "og:description",
        content: "Hierarchical archive of past gait scans with temporal attention previews.",
      },
    ],
  }),
  component: HistoryPage,
});

/** Patient currently held in the global header context. */
const activePatient = { id: "#PX-80492", name: "Arthur Bennett" };

type Leaf = {
  id: string;
  label: string;
  patient: string;
  date: string;
  dx: string;
  confidence: number;
  tone: "normal" | "warn" | "risk";
  localization: string;
  spatial: string;
  summary: string;
};

const tree: { year: string; months: { name: string; scans: Leaf[] }[] }[] = [
  {
    year: "2026",
    months: [
      {
        name: "July",
        scans: [
          {
            id: "s1",
            label: "12 Jul · Spastic Gait",
            patient: `${activePatient.id} · ${activePatient.name}`,
            date: "12 Jul 2026 · 09:41",
            dx: "Spastic",
            confidence: 0.78,
            tone: "risk",
            localization: "Right knee & ankle dorsiflexion",
            spatial: "Pose graph — right knee joint, edge R-hip→R-knee",
            summary:
              "Sustained attention peaks during terminal stance with reduced knee flexion amplitude on the right limb. Circumduction pattern detected across 3 consecutive cycles.",
          },
          {
            id: "s2",
            label: "03 Jul · Follow-up",
            patient: `${activePatient.id} · ${activePatient.name}`,
            date: "03 Jul 2026 · 11:05",
            dx: "Spastic",
            confidence: 0.71,
            tone: "warn",
            localization: "Right ankle push-off deficit",
            spatial: "Pose graph — right ankle joint, edge R-knee→R-ankle",
            summary:
              "Mild improvement in stance symmetry following 4 weeks of physiotherapy. Attention still concentrated on the right ankle during push-off.",
          },
        ],
      },
      {
        name: "May",
        scans: [
          {
            id: "s3",
            label: "14 May · Baseline",
            patient: `${activePatient.id} · ${activePatient.name}`,
            date: "14 May 2026 · 08:30",
            dx: "Spastic",
            confidence: 0.66,
            tone: "warn",
            localization: "Bilateral knee flexion asymmetry",
            spatial: "Pose graph — bilateral knees, trunk edge cluster",
            summary:
              "Baseline post-operative assessment. Cadence 96 steps/min, symmetry index 0.83.",
          },
        ],
      },
    ],
  },
  {
    year: "2025",
    months: [
      {
        name: "December",
        scans: [
          {
            id: "s4",
            label: "02 Dec · Initial Scan",
            patient: `${activePatient.id} · ${activePatient.name}`,
            date: "02 Dec 2025 · 13:47",
            dx: "Normal",
            confidence: 0.9,
            tone: "normal",
            localization: "No focal abnormality",
            spatial: "Pose graph — uniform joint saliency",
            summary:
              "Pre-injury reference walk test. Attention distributed evenly across heel strike and midstance.",
          },
        ],
      },
    ],
  },
];

const toneClass = {
  normal: "bg-normal text-normal-foreground",
  warn: "bg-warn text-warn-foreground",
  risk: "bg-risk text-risk-foreground",
};

function HistoryPage() {
  const [openYears, setOpenYears] = useState<string[]>(["2026"]);
  const [openMonths, setOpenMonths] = useState<string[]>(["2026July"]);
  const [selected, setSelected] = useState<Leaf>(tree[0]!.months[0]!.scans[0]!);

  const toggle = (arr: string[], set: (v: string[]) => void, key: string) =>
    set(arr.includes(key) ? arr.filter((k) => k !== key) : [...arr, key]);

  return (
    <AppShell>
      <div className="mx-auto max-w-[1400px] px-4 py-6 md:px-8">
        <div className="mb-6">
          <p className="text-xs font-medium tracking-wide text-primary uppercase">Archive</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight">Analytical History</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            4 archived sessions for {activePatient.id} · grouped by year and month
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-[minmax(300px,380px)_1fr]">
          <section className="clinical-card p-3">
            <div className="flex items-center justify-between px-2 py-2">
              <p className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
                Session tree
              </p>
              <span className="rounded-full bg-primary-soft px-2 py-0.5 text-[11px] font-medium text-accent-foreground">
                Filtered · {activePatient.id}
              </span>
            </div>
            <div className="space-y-0.5">
              {tree.map((y) => {
                const yOpen = openYears.includes(y.year);
                return (
                  <div key={y.year}>
                    <button
                      onClick={() => toggle(openYears, setOpenYears, y.year)}
                      className="flex w-full items-center gap-2 rounded-lg px-2 py-2 text-sm font-medium hover:bg-muted"
                    >
                      <ChevronRight
                        className={cn("size-3.5 text-muted-foreground transition-transform", yOpen && "rotate-90")}
                      />
                      {yOpen ? (
                        <FolderOpen className="size-4 text-primary" />
                      ) : (
                        <Folder className="size-4 text-primary" />
                      )}
                      {y.year}
                    </button>

                    {yOpen &&
                      y.months.map((q) => {
                        const qKey = y.year + q.name;
                        const qOpen = openMonths.includes(qKey);
                        return (
                          <div key={qKey} className="ml-4 border-l border-border pl-2">
                            <button
                              onClick={() => toggle(openMonths, setOpenMonths, qKey)}
                              className="flex w-full items-center gap-2 rounded-lg px-2 py-2 text-sm hover:bg-muted"
                            >
                              <ChevronRight
                                className={cn(
                                  "size-3.5 text-muted-foreground transition-transform",
                                  qOpen && "rotate-90",
                                )}
                              />
                              {qOpen ? (
                                <FolderOpen className="size-4 text-muted-foreground" />
                              ) : (
                                <Folder className="size-4 text-muted-foreground" />
                              )}
                              <span className="truncate">{q.name}</span>
                              <span className="ml-auto text-[11px] text-muted-foreground">
                                {q.scans.length}
                              </span>
                            </button>

                            {qOpen && (
                              <div className="ml-4 border-l border-border pl-2">
                                {q.scans.map((s) => (
                                  <button
                                    key={s.id}
                                    onClick={() => setSelected(s)}
                                    className={cn(
                                      "flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left text-sm transition-colors hover:bg-muted",
                                      selected.id === s.id &&
                                        "bg-primary-soft font-medium text-accent-foreground",
                                    )}
                                  >
                                    <FileText className="size-4 shrink-0 text-muted-foreground" />
                                    <span className="truncate">{s.label}</span>
                                  </button>
                                ))}
                              </div>
                            )}
                          </div>
                        );
                      })}
                  </div>
                );
              })}
            </div>
          </section>

          <section className="clinical-card p-5">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold">{selected.patient}</h2>
                <p className="mt-1 text-xs text-muted-foreground">{selected.date}</p>
              </div>
              <span
                className={cn("rounded-full px-3 py-1 text-xs font-medium", toneClass[selected.tone])}
              >
                {selected.dx} · {(selected.confidence * 100).toFixed(0)}%
              </span>
            </div>

            <div className="mt-5 rounded-xl border border-border bg-muted/30 p-3">
              <p className="mb-1 text-xs font-medium text-muted-foreground">
                Temporal attention preview
              </p>
              <TemporalAttentionChart axisLabels />
            </div>

            <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="rounded-xl border border-border p-4">
                <p className="text-xs font-medium text-muted-foreground">Clinical localization</p>
                <p className="mt-1.5 text-sm font-medium">{selected.localization}</p>
              </div>
              <div className="rounded-xl border border-border p-4">
                <p className="text-xs font-medium text-muted-foreground">
                  Spatial Localization (Pose/Edge)
                </p>
                <p className="mt-1.5 text-sm font-medium">{selected.spatial}</p>
                <div className="mt-3 flex h-20 items-center justify-center rounded-lg border border-dashed border-border bg-muted/40">
                  <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                    <Waypoints className="size-3.5" /> Pose graph overlay renders here
                  </span>
                </div>
              </div>
              <div className="rounded-xl border border-border p-4 sm:col-span-2">
                <p className="text-xs font-medium text-muted-foreground">Model</p>
                <p className="mt-1.5 text-sm font-medium">
                  GaitVision v2.4 · 32-frame sequence @ 30 fps
                </p>
              </div>
            </div>

            <div className="mt-4 rounded-xl border border-border p-4">
              <p className="text-xs font-medium text-muted-foreground">Quick summary</p>
              <p className="mt-1.5 text-sm leading-relaxed">{selected.summary}</p>
            </div>

            <div className="mt-5 flex gap-2">
              <Button size="sm">
                Open full analysis <ArrowUpRight className="size-4" />
              </Button>
              <Button size="sm" variant="outline">
                Export report
              </Button>
            </div>
          </section>
        </div>
      </div>
    </AppShell>
  );
}
