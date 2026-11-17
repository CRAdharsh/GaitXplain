import { createFileRoute } from "@tanstack/react-router";
import { ChevronDown, PlusCircle, Search, UserRound } from "lucide-react";
import { useMemo, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/patients")({
  head: () => ({
    meta: [
      { title: "Patient Directory — GaitVision AI" },
      {
        name: "description",
        content:
          "Ward and bedside patient directory with last gait scan dates and AI abnormality classification badges.",
      },
      { property: "og:title", content: "Patient Directory — GaitVision AI" },
      {
        property: "og:description",
        content: "Scannable clinical directory of patients, wards, beds and latest gait AI results.",
      },
    ],
  }),
  component: PatientsPage,
});

type Status = "Normal" | "Parkinsonian" | "Hemiplegic" | "Spastic" | "Pending";

const statusTone: Record<Status, string> = {
  Normal: "bg-normal text-normal-foreground",
  Parkinsonian: "bg-warn text-warn-foreground",
  Hemiplegic: "bg-risk text-risk-foreground",
  Spastic: "bg-risk text-risk-foreground",
  Pending: "bg-info text-info-foreground",
};

const patients: {
  id: string;
  name: string;
  age: number;
  sex: "M" | "F";
  ward: string;
  lastScan: string;
  status: Status;
}[] = [
  { id: "#PX-80492", name: "Arthur Bennett", age: 58, sex: "M", ward: "Neurology · Bed 4", lastScan: "12 Jul 2026", status: "Hemiplegic" },
  { id: "#PX-80493", name: "Marta Oyelaran", age: 44, sex: "F", ward: "Neurology · Bed 5", lastScan: "12 Jul 2026", status: "Normal" },
  { id: "#PX-80488", name: "Henrik Sørensen", age: 71, sex: "M", ward: "Geriatrics · Bed 12", lastScan: "11 Jul 2026", status: "Parkinsonian" },
  { id: "#PX-80471", name: "Claudia Ferrer", age: 33, sex: "F", ward: "Rehab Unit · Bed 2", lastScan: "09 Jul 2026", status: "Spastic" },
  { id: "#PX-80465", name: "Ravi Chandrasekar", age: 62, sex: "M", ward: "Neurology · Bed 9", lastScan: "08 Jul 2026", status: "Normal" },
  { id: "#PX-80459", name: "Eileen Park", age: 67, sex: "F", ward: "Geriatrics · Bed 3", lastScan: "07 Jul 2026", status: "Parkinsonian" },
  { id: "#PX-80440", name: "Tomasz Wójcik", age: 51, sex: "M", ward: "Rehab Unit · Bed 8", lastScan: "05 Jul 2026", status: "Pending" },
  { id: "#PX-80431", name: "Nadia Haddad", age: 39, sex: "F", ward: "Orthopedics · Bed 6", lastScan: "03 Jul 2026", status: "Normal" },
];

function PatientsPage() {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("all");

  const rows = useMemo(
    () =>
      patients.filter((p) => {
        const q = query.trim().toLowerCase();
        const matchQ = !q || p.name.toLowerCase().includes(q) || p.id.toLowerCase().includes(q);
        const matchS = status === "all" || p.status === status;
        return matchQ && matchS;
      }),
    [query, status],
  );

  return (
    <AppShell>
      <div className="mx-auto max-w-[1400px] px-4 py-6 md:px-8">
        <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="text-xs font-medium tracking-wide text-primary uppercase">Ward directory</p>
            <h1 className="mt-1 text-2xl font-semibold tracking-tight">Patient Directory</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              {rows.length} of {patients.length} admitted patients · last synced 06:12 today
            </p>
          </div>
          <Button size="sm">
            <PlusCircle className="size-4" /> Admit Patient
          </Button>
        </div>

        <section className="clinical-card overflow-hidden">
          <div className="flex flex-wrap items-center gap-3 border-b border-border p-4">
            <div className="relative min-w-56 flex-1">
              <Search className="absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search by ID or name"
                className="pl-9"
              />
            </div>
            <Select value={status} onValueChange={setStatus}>
              <SelectTrigger className="w-52">
                <SelectValue placeholder="Filter by status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">Filter by Status: All</SelectItem>
                <SelectItem value="Normal">Normal</SelectItem>
                <SelectItem value="Parkinsonian">Parkinsonian</SelectItem>
                <SelectItem value="Hemiplegic">Hemiplegic</SelectItem>
                <SelectItem value="Spastic">Spastic</SelectItem>
                <SelectItem value="Pending">Pending review</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full min-w-[880px] text-sm">
              <thead>
                <tr className="border-b border-border bg-muted/50 text-left text-xs font-medium tracking-wide text-muted-foreground uppercase">
                  <th className="px-4 py-3">Patient ID</th>
                  <th className="px-4 py-3">Full Name</th>
                  <th className="px-4 py-3">Age / Sex</th>
                  <th className="px-4 py-3">Ward / Bed</th>
                  <th className="px-4 py-3">Last Scan</th>
                  <th className="px-4 py-3">AI Status</th>
                  <th className="px-4 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((p) => (
                  <tr key={p.id} className="border-b border-border last:border-0 hover:bg-muted/40">
                    <td className="px-4 py-3 font-mono text-xs text-muted-foreground">{p.id}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2.5">
                        <span className="flex size-7 items-center justify-center rounded-full bg-primary-soft text-[11px] font-semibold text-accent-foreground">
                          {p.name
                            .split(" ")
                            .map((n) => n[0])
                            .join("")}
                        </span>
                        <span className="font-medium">{p.name}</span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">
                      {p.age} · {p.sex}
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">{p.ward}</td>
                    <td className="px-4 py-3 text-muted-foreground">{p.lastScan}</td>
                    <td className="px-4 py-3">
                      <span
                        className={cn(
                          "inline-flex rounded-full px-2.5 py-1 text-xs font-medium",
                          statusTone[p.status],
                        )}
                      >
                        {p.status}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex justify-end gap-1.5">
                        <Button variant="ghost" size="sm" className="text-muted-foreground">
                          <UserRound className="size-4" /> View Profile
                        </Button>
                        <Button variant="outline" size="sm">
                          New Scan
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
                {rows.length === 0 && (
                  <tr>
                    <td colSpan={7} className="px-4 py-10 text-center text-sm text-muted-foreground">
                      No patients match the current filters.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="flex items-center justify-between border-t border-border px-4 py-3 text-xs text-muted-foreground">
            <span>Showing {rows.length} records</span>
            <button className="inline-flex items-center gap-1 hover:text-primary">
              Load older admissions <ChevronDown className="size-3.5" />
            </button>
          </div>
        </section>
      </div>
    </AppShell>
  );
}
