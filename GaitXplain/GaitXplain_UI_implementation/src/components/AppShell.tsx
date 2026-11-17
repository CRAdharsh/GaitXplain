import { Link, useRouterState } from "@tanstack/react-router";
import {
  Activity,
  ChevronLeft,
  FileDown,
  History,
  Gauge,
  Printer,
  Settings,
  Users,
  Video,
} from "lucide-react";
import { useState, type ReactNode } from "react";

import { GaitIcon } from "./GaitIcon";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

const navItems = [
  { title: "Patients List", url: "/patients", icon: Users },
  { title: "New Gait Scan", url: "/", icon: Video },
  { title: "Analytical History", url: "/history", icon: History },
  { title: "System Metrics", url: "/metrics", icon: Gauge },
  { title: "Settings", url: "/settings", icon: Settings },
];

/** Routes that operate on a single patient keep the patient context bar. */
const patientRoutes = ["/", "/analysis", "/history"];

const breadcrumbs: Record<string, string[]> = {
  "/metrics": ["Admin", "System Metrics"],
  "/settings": ["Admin", "Settings"],
  "/patients": ["Clinic", "Patients List"],
};



export function AppShell({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const showPatientContext = patientRoutes.includes(pathname);
  const crumbs = breadcrumbs[pathname] ?? ["Admin", "Overview"];

  return (
    <TooltipProvider delayDuration={120}>
      <div className="min-h-screen w-full bg-background">
        <header className="sticky top-0 z-30 flex h-16 items-center gap-4 border-b border-border bg-surface px-4 md:px-6">
          <div className="flex items-center gap-2.5">
            <div className="flex size-9 items-center justify-center rounded-lg bg-primary text-primary-foreground">
              <GaitIcon className="size-5" />
            </div>
            <div className="leading-tight">
              <p className="text-sm font-semibold tracking-tight">GaitVision AI</p>
              <p className="text-[11px] text-muted-foreground">Clinical Assessment</p>
            </div>
          </div>

          <Separator orientation="vertical" className="hidden h-8 lg:block" />

          {showPatientContext ? (
            <div className="hidden items-center gap-3 rounded-lg border border-border bg-muted/60 px-3 py-1.5 lg:flex">
              <span className="size-2 rounded-full bg-normal-foreground" />
              <span className="text-xs font-medium">Patient ID: #PX-80492</span>
              <span className="text-xs text-muted-foreground">Age: 58</span>
              <span className="text-xs text-muted-foreground">Sex: M</span>
            </div>
          ) : (
            <nav aria-label="Breadcrumb" className="hidden items-center gap-1.5 lg:flex">
              {crumbs.map((c, i) => (
                <span key={c} className="flex items-center gap-1.5">
                  {i > 0 && <span className="text-xs text-muted-foreground">/</span>}
                  <span
                    className={cn(
                      "text-xs",
                      i === crumbs.length - 1
                        ? "font-medium text-foreground"
                        : "text-muted-foreground",
                    )}
                  >
                    {c}
                  </span>
                </span>
              ))}
            </nav>
          )}

          <div className="ml-auto flex items-center gap-2">
            <Button variant="outline" size="sm" className="hidden sm:inline-flex">
              <FileDown className="size-4" /> PDF Export
            </Button>
            <Button variant="outline" size="sm" className="hidden md:inline-flex">
              <Printer className="size-4" /> Print Report
            </Button>
            <div className="flex items-center gap-2 rounded-full border border-border py-1 pr-3 pl-1">
              <span className="flex size-7 items-center justify-center rounded-full bg-primary-soft text-xs font-semibold text-accent-foreground">
                DS
              </span>
              <span className="hidden text-xs font-medium sm:block">Dr. Specialist</span>
            </div>
          </div>
        </header>

        <div className="flex">
          <aside
            className={cn(
              "sticky top-16 hidden h-[calc(100vh-4rem)] shrink-0 border-r border-sidebar-border bg-sidebar transition-[width] duration-200 md:block",
              collapsed ? "w-16" : "w-60",
            )}
          >
            <nav className="flex h-full flex-col gap-1 p-3">
              {navItems.map((item) => {
                const active =
                  item.url === "/"
                    ? pathname === "/" || pathname === "/analysis"
                    : pathname === item.url || pathname.startsWith(item.url + "/");
                const link = (
                  <Link
                    key={item.title}
                    to={item.url}
                    aria-current={active ? "page" : undefined}
                    className={cn(
                      "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-sidebar-foreground transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                      active && "bg-sidebar-accent text-sidebar-accent-foreground",
                      collapsed && "justify-center px-0",
                    )}
                  >
                    <item.icon className="size-4.5 shrink-0" />
                    {!collapsed && <span className="truncate">{item.title}</span>}
                  </Link>
                );
                return collapsed ? (
                  <Tooltip key={item.title}>
                    <TooltipTrigger asChild>{link}</TooltipTrigger>
                    <TooltipContent side="right">{item.title}</TooltipContent>
                  </Tooltip>
                ) : (
                  link
                );
              })}

              <div className="mt-auto shrink-0 pt-3">
                {!collapsed && (
                  <div className="mb-3 rounded-lg border border-border bg-muted/50 p-3">
                    <p className="flex items-center gap-1.5 text-xs font-medium">
                      <Activity className="size-3.5 text-primary" /> Model v2.4 · online
                    </p>
                    <p className="mt-1 text-[11px] text-muted-foreground">
                      Inference latency 1.8s / 224×224 clip
                    </p>
                  </div>
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full justify-center"
                  onClick={() => setCollapsed((c) => !c)}
                >
                  <ChevronLeft className={cn("size-4 transition-transform", collapsed && "rotate-180")} />
                  {!collapsed && <span>Collapse</span>}
                </Button>
              </div>
            </nav>
          </aside>

          <main className="min-w-0 flex-1">{children}</main>
        </div>
      </div>
    </TooltipProvider>
  );
}
