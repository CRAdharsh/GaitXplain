export function GaitIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" fill="none" className={className} aria-hidden="true">
      <circle cx="17" cy="5" r="3" fill="currentColor" />
      <path
        d="M17 9v7m0 0l-6 6m6-6l5 5m-11 1l-2 6m13-5l2 6"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M10 13l7 2" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" opacity="0.5" />
    </svg>
  );
}
