import { createFileRoute } from "@tanstack/react-router";
import { KeyRound, ShieldCheck, SlidersHorizontal, Upload, UserRound } from "lucide-react";
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
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/settings")({
  head: () => ({
    meta: [
      { title: "Settings — GaitVision AI" },
      {
        name: "description",
        content:
          "Manage clinician profile, application preferences, export defaults and account security for GaitVision AI.",
      },
      { property: "og:title", content: "Settings — GaitVision AI" },
      {
        property: "og:description",
        content: "Clinician profile, preferences and security settings for the gait analysis platform.",
      },
    ],
  }),
  component: SettingsPage,
});

const tabs = [
  { id: "profile", label: "My Profile", icon: UserRound, hint: "Name, specialization, avatar" },
  { id: "prefs", label: "Preferences", icon: SlidersHorizontal, hint: "Theme, alerts, exports" },
  { id: "security", label: "Security", icon: ShieldCheck, hint: "Password & sessions" },
] as const;

function Row({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-6 py-4">
      <div>
        <p className="text-sm font-medium">{title}</p>
        <p className="mt-0.5 text-xs text-muted-foreground">{description}</p>
      </div>
      {children}
    </div>
  );
}

function SettingsPage() {
  const [tab, setTab] = useState<string>("profile");
  const [darkMode, setDarkMode] = useState(false);
  const [alerts, setAlerts] = useState(true);
  const [autoArchive, setAutoArchive] = useState(false);

  return (
    <AppShell>
      <div className="mx-auto max-w-[1200px] px-4 py-6 md:px-8">
        <div className="mb-6">
          <p className="text-xs font-medium tracking-wide text-primary uppercase">Account</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight">Settings</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Manage your clinician profile, application behaviour and account security.
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-[minmax(220px,260px)_1fr]">
          <nav className="clinical-card h-fit p-2">
            {tabs.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={cn(
                  "flex w-full items-start gap-3 rounded-lg px-3 py-2.5 text-left transition-colors hover:bg-muted",
                  tab === t.id && "bg-primary-soft text-accent-foreground",
                )}
              >
                <t.icon className="mt-0.5 size-4 shrink-0" />
                <span className="min-w-0">
                  <span className="block text-sm font-medium">{t.label}</span>
                  <span className="block truncate text-[11px] text-muted-foreground">{t.hint}</span>
                </span>
              </button>
            ))}
          </nav>

          <section className="clinical-card p-6">
            {tab === "profile" && (
              <div>
                <h2 className="text-sm font-semibold">My Profile</h2>
                <p className="mt-1 text-xs text-muted-foreground">
                  Shown on exported clinical reports and physician remarks.
                </p>

                <div className="mt-6 flex flex-wrap items-center gap-4">
                  <span className="flex size-16 items-center justify-center rounded-full bg-primary-soft text-lg font-semibold text-accent-foreground">
                    DS
                  </span>
                  <div>
                    <Button variant="outline" size="sm">
                      <Upload className="size-4" /> Upload profile picture
                    </Button>
                    <p className="mt-1.5 text-[11px] text-muted-foreground">
                      PNG or JPG, square, up to 2 MB.
                    </p>
                  </div>
                </div>

                <div className="mt-6 grid gap-5 sm:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="name">Full name</Label>
                    <Input id="name" defaultValue="Dr. Amara Specialist" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="spec">Specialization</Label>
                    <Input id="spec" defaultValue="Neurology · Movement Disorders" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="email">Email address</Label>
                    <Input id="email" type="email" defaultValue="a.specialist@stmartins-health.org" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="lic">Medical license ID</Label>
                    <Input id="lic" defaultValue="MD-441-2098" />
                  </div>
                </div>
              </div>
            )}

            {tab === "prefs" && (
              <div>
                <h2 className="text-sm font-semibold">Preferences</h2>
                <p className="mt-1 text-xs text-muted-foreground">
                  Control the interface and how results reach you.
                </p>

                <div className="mt-4 divide-y divide-border">
                  <Row title="Enable Dark Mode" description="Reduced-luminance theme for reading rooms.">
                    <Switch checked={darkMode} onCheckedChange={setDarkMode} />
                  </Row>
                  <Row
                    title="Receive Email Alerts for Severe Diagnoses"
                    description="Notify me when confidence for a high-risk class exceeds 70%."
                  >
                    <Switch checked={alerts} onCheckedChange={setAlerts} />
                  </Row>
                  <Row
                    title="Auto-archive raw RGB videos"
                    description="Move source clips to cold storage after 90 days."
                  >
                    <Switch checked={autoArchive} onCheckedChange={setAutoArchive} />
                  </Row>
                  <Row
                    title="Default Export Format"
                    description="Applied to report downloads from the diagnostic dashboard."
                  >
                    <Select defaultValue="pdf">
                      <SelectTrigger className="w-40">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="pdf">PDF report</SelectItem>
                        <SelectItem value="csv">CSV metrics</SelectItem>
                      </SelectContent>
                    </Select>
                  </Row>
                </div>
              </div>
            )}

            {tab === "security" && (
              <div>
                <h2 className="text-sm font-semibold">Security</h2>
                <p className="mt-1 text-xs text-muted-foreground">
                  Credentials and recent account activity.
                </p>

                <div className="mt-6 grid max-w-md gap-5">
                  <div className="space-y-2">
                    <Label htmlFor="current">Current password</Label>
                    <Input id="current" type="password" placeholder="••••••••••" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="new">New password</Label>
                    <Input id="new" type="password" placeholder="At least 12 characters" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="confirm">Confirm new password</Label>
                    <Input id="confirm" type="password" placeholder="Re-enter new password" />
                  </div>
                  <Button variant="outline" size="sm" className="w-fit">
                    <KeyRound className="size-4" /> Update password
                  </Button>
                </div>

                <Separator className="my-6" />

                <div className="rounded-xl border border-border bg-muted/40 p-4">
                  <p className="text-xs font-medium text-muted-foreground">Last login</p>
                  <p className="mt-1 text-sm font-medium">03 Aug 2026 · 06:12 UTC</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Chrome on macOS · 10.4.22.18 · Hospital network
                  </p>
                </div>
              </div>
            )}

            <Separator className="my-6" />

            <div className="flex items-center justify-between gap-3">
              <p className="text-xs text-muted-foreground">Changes apply to this workstation only.</p>
              <div className="flex gap-2">
                <Button variant="ghost" size="sm">
                  Cancel
                </Button>
                <Button size="sm">Save Changes</Button>
              </div>
            </div>
          </section>
        </div>
      </div>
    </AppShell>
  );
}
