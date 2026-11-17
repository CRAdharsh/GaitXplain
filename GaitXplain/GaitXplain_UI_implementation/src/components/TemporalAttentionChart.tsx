import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from "recharts";

import { cn } from "@/lib/utils";

export const gaitPhases = [
  { name: "Heel Strike", start: 0, end: 14, color: "oklch(0.95 0.05 160)" },
  { name: "Loading", start: 14, end: 32, color: "oklch(0.96 0.04 220)" },
  { name: "Midstance", start: 32, end: 56, color: "oklch(0.95 0.05 25)" },
  { name: "Terminal Stance", start: 56, end: 78, color: "oklch(0.96 0.06 85)" },
  { name: "Swing Phase", start: 78, end: 100, color: "oklch(0.96 0.03 300)" },
];

const data = Array.from({ length: 101 }, (_, i) => {
  const base =
    0.18 +
    0.55 * Math.exp(-((i - 46) ** 2) / 220) +
    0.22 * Math.exp(-((i - 86) ** 2) / 160) +
    0.06 * Math.sin(i / 4);
  return { frame: i, score: Number(Math.max(0.04, Math.min(1, base)).toFixed(3)) };
});

export function TemporalAttentionChart({ axisLabels = false }: { axisLabels?: boolean }) {
  return (
    <div className="relative h-56 w-full">
      <div
        className={cn(
          "pointer-events-none absolute top-2 right-2 z-0 flex overflow-hidden rounded-sm",
          axisLabels ? "bottom-12 left-14" : "bottom-7 left-8",
        )}
      >
        {gaitPhases.map((p) => (
          <div
            key={p.name}
            style={{ width: `${p.end - p.start}%`, background: p.color }}
            className="h-full opacity-80"
          />
        ))}
      </div>
      <div className="relative z-10 h-full w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={data}
          margin={
            axisLabels
              ? { top: 8, right: 8, left: 4, bottom: 12 }
              : { top: 8, right: 8, left: -18, bottom: 0 }
          }
        >
          <defs>
            <linearGradient id="attn" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--color-primary)" stopOpacity={0.4} />
              <stop offset="100%" stopColor="var(--color-primary)" stopOpacity={0.03} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
          <XAxis
            dataKey="frame"
            tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }}
            tickLine={false}
            axisLine={{ stroke: "var(--color-border)" }}
            label={
              axisLabels
                ? {
                    value: "Frame Index",
                    position: "insideBottom",
                    offset: -8,
                    style: { fontSize: 11, fill: "var(--color-muted-foreground)" },
                  }
                : undefined
            }
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }}
            tickLine={false}
            axisLine={false}
            label={
              axisLabels
                ? {
                    value: "Attention Weight",
                    angle: -90,
                    position: "insideLeft",
                    offset: 8,
                    style: {
                      fontSize: 11,
                      fill: "var(--color-muted-foreground)",
                      textAnchor: "middle",
                    },
                  }
                : undefined
            }
          />
          <RTooltip
            contentStyle={{
              borderRadius: 10,
              border: "1px solid var(--color-border)",
              fontSize: 12,
              boxShadow: "var(--shadow-clinical-md)",
            }}
            formatter={(v: number) => [v, "Attention"]}
            labelFormatter={(l) => `Frame ${l}`}
          />
          <Area
            type="monotone"
            dataKey="score"
            stroke="var(--color-primary)"
            strokeWidth={2}
            fill="url(#attn)"
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
      </div>
    </div>
  );
}
