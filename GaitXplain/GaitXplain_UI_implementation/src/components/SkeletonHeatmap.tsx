import { cn } from "@/lib/utils";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

type Joint = { id: string; label: string; x: number; y: number; score: number };

export const joints: Joint[] = [
  { id: "head", label: "Head / Cervical", x: 100, y: 26, score: 0.08 },
  { id: "torso", label: "Torso", x: 100, y: 82, score: 0.21 },
  { id: "hipL", label: "Left Hip", x: 82, y: 118, score: 0.26 },
  { id: "hipR", label: "Right Hip", x: 118, y: 118, score: 0.54 },
  { id: "kneeL", label: "Left Knee", x: 78, y: 176, score: 0.31 },
  { id: "kneeR", label: "Right Knee", x: 124, y: 176, score: 0.92 },
  { id: "ankleL", label: "Left Ankle", x: 74, y: 232, score: 0.29 },
  { id: "ankleR", label: "Right Ankle", x: 130, y: 232, score: 0.81 },
  { id: "footL", label: "Left Foot", x: 70, y: 258, score: 0.18 },
  { id: "footR", label: "Right Foot", x: 136, y: 258, score: 0.63 },
];

const bones: [string, string][] = [
  ["head", "torso"],
  ["torso", "hipL"],
  ["torso", "hipR"],
  ["hipL", "kneeL"],
  ["hipR", "kneeR"],
  ["kneeL", "ankleL"],
  ["kneeR", "ankleR"],
  ["ankleL", "footL"],
  ["ankleR", "footR"],
];

function heatColor(score: number) {
  // blue (240deg) -> red (0deg)
  const hue = 235 - score * 235;
  return `hsl(${hue} 82% 55%)`;
}

export function SkeletonHeatmap() {
  const byId = new Map(joints.map((j) => [j.id, j]));

  return (
    <div>
      <svg viewBox="0 0 200 285" className="mx-auto h-72 w-full">
        {bones.map(([a, b]) => {
          const ja = byId.get(a);
          const jb = byId.get(b);
          if (!ja || !jb) return null;
          return (
            <line
              key={`${a}-${b}`}
              x1={ja.x}
              y1={ja.y}
              x2={jb.x}
              y2={jb.y}
              stroke="currentColor"
              className="text-border"
              strokeWidth={3}
              strokeLinecap="round"
            />
          );
        })}
        {joints.map((j) => (
          <Tooltip key={j.id}>
            <TooltipTrigger asChild>
              <g className="cursor-pointer">
                <circle cx={j.x} cy={j.y} r={16} fill={heatColor(j.score)} opacity={0.18} />
                <circle
                  cx={j.x}
                  cy={j.y}
                  r={7 + j.score * 4}
                  fill={heatColor(j.score)}
                  className="transition-all duration-300"
                />
              </g>
            </TooltipTrigger>
            <TooltipContent>
              {j.label} · contribution {(j.score * 100).toFixed(0)}%
            </TooltipContent>
          </Tooltip>
        ))}
      </svg>

      <div className="mt-2">
        <div
          className="h-2 w-full rounded-full"
          style={{
            background:
              "linear-gradient(90deg, hsl(235 82% 55%), hsl(180 82% 45%), hsl(60 85% 50%), hsl(0 82% 55%))",
          }}
        />
        <div className="mt-1 flex justify-between text-[11px] text-muted-foreground">
          <span>Low contribution</span>
          <span>High abnormality</span>
        </div>
      </div>

      <ul className="mt-4 space-y-1.5">
        {[...joints]
          .sort((a, b) => b.score - a.score)
          .slice(0, 4)
          .map((j) => (
            <li key={j.id} className="flex items-center gap-2 text-xs">
              <span className="size-2 rounded-full" style={{ background: heatColor(j.score) }} />
              <span className={cn("flex-1", j.score > 0.7 && "font-medium")}>{j.label}</span>
              <span className="font-mono text-muted-foreground">{(j.score * 100).toFixed(0)}%</span>
            </li>
          ))}
      </ul>
    </div>
  );
}
