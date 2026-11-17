import { createFileRoute, Link } from "@tanstack/react-router";
import {
  CircleAlert,
  FileVideo,
  Info,
  Loader2,
  Pause,
  Play,
  Sparkles,
  UploadCloud,
} from "lucide-react";
import { useState } from "react";

import { AppShell } from "@/components/AppShell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "GaitVision AI Clinical Assessment" },
      {
        name: "description",
        content:
          "Upload RGB gait videos and capture patient intake metadata for explainable gait abnormality localization.",
      },
      { property: "og:title", content: "GaitVision AI Clinical Assessment" },
      {
        property: "og:description",
        content:
          "Upload RGB gait videos and capture patient intake metadata for explainable gait abnormality localization.",
      },
    ],
  }),
  component: IntakePage,
});

function ClinicalTip({ text }: { text: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button type="button" className="text-muted-foreground transition-colors hover:text-primary">
          <Info className="size-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent className="max-w-64">{text}</TooltipContent>
    </Tooltip>
  );
}

function IntakePage() {
  const [running, setRunning] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [frame, setFrame] = useState([42]);

  return (
    <AppShell>
      <div className="mx-auto max-w-7xl px-4 py-6 md:px-8">
        <div className="mb-6">
          <p className="text-xs font-medium tracking-wide text-primary uppercase">Step 1 of 2</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight">
            Gait Video Upload &amp; Patient Intake
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Explainable Gait Abnormality Localization System — capture clinical context before running
            inference.
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          {/* Left: patient details */}
          <section className="clinical-card p-5">
            <h2 className="text-sm font-semibold">Patient Details &amp; Metadata</h2>
            <p className="mt-1 text-xs text-muted-foreground">
              All fields are stored with the assessment record.
            </p>

            <div className="mt-5 grid gap-4 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label htmlFor="pid">Patient ID</Label>
                <Input id="pid" defaultValue="PX-80492" />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="age">Age</Label>
                <Input id="age" type="number" defaultValue={58} />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="sex">Gender</Label>
                <Select defaultValue="male">
                  <SelectTrigger id="sex">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="male">Male</SelectItem>
                    <SelectItem value="female">Female</SelectItem>
                    <SelectItem value="other">Other / Not specified</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="cond" className="gap-1.5">
                  Walk Test Condition
                  <ClinicalTip text="Treadmill trials normalize cadence; open-hallway 10MWT better captures turning and free-speed asymmetry." />
                </Label>
                <Select defaultValue="10mwt">
                  <SelectTrigger id="cond">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="10mwt">Open Hallway — 10-meter Walk Test</SelectItem>
                    <SelectItem value="treadmill">Treadmill — fixed speed</SelectItem>
                    <SelectItem value="tug">Timed Up &amp; Go</SelectItem>
                    <SelectItem value="free">Free-speed corridor walk</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5 sm:col-span-2">
                <Label htmlFor="concern" className="gap-1.5">
                  Primary Observation Concern
                  <ClinicalTip text="Guides the report narrative only. Classification remains unbiased across all six target classes." />
                </Label>
                <Select defaultValue="stroke">
                  <SelectTrigger id="concern">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="stroke">Post-Stroke Hemiparesis</SelectItem>
                    <SelectItem value="parkinson">Parkinsonian Tremor / Bradykinesia</SelectItem>
                    <SelectItem value="ataxia">General Ataxia</SelectItem>
                    <SelectItem value="spastic">Spastic Diplegia</SelectItem>
                    <SelectItem value="neuro">Peripheral Neuropathy</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5 sm:col-span-2">
                <Label htmlFor="notes">Primary Clinical Notes</Label>
                <Textarea
                  id="notes"
                  rows={5}
                  defaultValue="Patient reports right-sided weakness following CVA (11 months prior). Ambulates without assistive device, mild circumduction observed on visual inspection."
                />
              </div>
            </div>

            <div className="mt-5 flex items-start gap-2 rounded-lg border border-border bg-info px-3 py-2.5">
              <CircleAlert className="mt-0.5 size-4 shrink-0 text-info-foreground" />
              <p className="text-xs text-info-foreground">
                De-identify footage before upload. Faces are automatically blurred during preprocessing.
              </p>
            </div>
          </section>

          {/* Right: dropzone */}
          <section className="clinical-card flex flex-col p-5">
            <h2 className="text-sm font-semibold">RGB Gait Video</h2>
            <p className="mt-1 text-xs text-muted-foreground">
              Sagittal or coronal capture, minimum 3 full gait cycles.
            </p>

            <label
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
              }}
              className={`mt-4 flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors ${
                dragging ? "border-primary bg-primary-soft" : "border-border bg-muted/40 hover:border-primary/50"
              }`}
            >
              <input type="file" accept=".mp4,.avi,.mov" className="sr-only" />
              <span className="flex size-11 items-center justify-center rounded-full bg-primary-soft text-primary">
                <UploadCloud className="size-5" />
              </span>
              <span className="mt-3 text-sm font-medium">Drag &amp; drop gait video here</span>
              <span className="mt-1 text-xs text-muted-foreground">
                or click to browse — .mp4, .avi, .mov · up to 500 MB
              </span>
            </label>

            {/* Preview player */}
            <div className="mt-4 rounded-xl border border-border p-3">
              <div className="flex items-center justify-between">
                <p className="flex items-center gap-2 text-xs font-medium">
                  <FileVideo className="size-4 text-primary" />
                  PX-80492_10mwt_sagittal.mp4
                </p>
                <span className="rounded-md bg-muted px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
                  T × 224 × 224
                </span>
              </div>

              <div className="mt-3 flex aspect-video items-center justify-center rounded-lg bg-linear-to-br from-slate-100 to-slate-200 text-xs text-muted-foreground">
                <span className="rounded-md bg-surface/80 px-2 py-1">Preview · 1920×1080 → 224×224</span>
              </div>

              <div className="mt-3 flex items-center gap-3">
                <Button
                  size="icon"
                  variant="secondary"
                  className="size-8 rounded-full"
                  onClick={() => setPlaying((p) => !p)}
                >
                  {playing ? <Pause className="size-4" /> : <Play className="size-4" />}
                </Button>
                <Slider value={frame} onValueChange={setFrame} max={120} step={1} className="flex-1" />
                <span className="font-mono text-[11px] text-muted-foreground">
                  {String(frame[0]).padStart(3, "0")}/120
                </span>
              </div>
            </div>

            {/* Toggles */}
            <div className="mt-4 space-y-2">
              {[
                {
                  label: "Include Joint-Level Heatmap Extraction",
                  desc: "Per-joint spatial abnormality contribution maps.",
                },
                {
                  label: "Generate Temporal Gait-Phase Breakdown",
                  desc: "Heel strike → swing segmentation with attention weights.",
                },
              ].map((t) => (
                <div
                  key={t.label}
                  className="flex items-center justify-between gap-4 rounded-lg border border-border px-3 py-2.5"
                >
                  <div>
                    <p className="text-xs font-medium">{t.label}</p>
                    <p className="text-[11px] text-muted-foreground">{t.desc}</p>
                  </div>
                  <Switch defaultChecked />
                </div>
              ))}
            </div>

            <Button
              size="lg"
              className="mt-5 w-full"
              onClick={() => setRunning(true)}
              disabled={running}
              asChild={false}
            >
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" /> Running inference…
                </>
              ) : (
                <>
                  <Sparkles className="size-4" /> Run Explainable Gait Analysis
                </>
              )}
            </Button>
            {running && (
              <Link
                to="/analysis"
                className="mt-3 text-center text-xs font-medium text-primary hover:underline"
              >
                View diagnostic output dashboard →
              </Link>
            )}
          </section>
        </div>
      </div>
    </AppShell>
  );
}
