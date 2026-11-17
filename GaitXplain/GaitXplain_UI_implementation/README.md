# Gait Insight

Create a modern, white-themed, medical-grade SaaS web dashboard interface in Tailwind CSS / Laravel Blade component style for a clinical application titled "Explainable Gait Abnormality Localization System". 

### 1. Visual Aesthetics & Design System

- Theme: Clean, high-contrast Light/White mode. Crisp clinical feel with subtle warm gray tones.

- Palette: 

  - Primary Background: #FFFFFF (Pure White) and #F8FAFC (Slate 50 background tint).

  - Borders & Cards: #E2E8F0 (Slate 200) with subtle, soft drop shadows (`shadow-sm` / `shadow-md`).

  - Primary Accent: #0284C7 (Medical Slate Blue) or #2563EB (Clinical Royal Blue).

  - Diagnostic Class Badges: Soft pastel backgrounds with dark colored text (e.g., Soft Green for Normal, Soft Amber for Parkinsonian, Soft Rose for Hemiplegic/Spastic).

- Typography: Inter or SF Pro Display — clean, highly readable, balanced font weights (Medium/Semibold for labels, Regular for clinical telemetry).

- Overall Vibe: Professional, calm, ergonomic for neurologists and physical therapists. Avoid dark neon AI tropes, glowing borders, or sci-fi graphics.

---

### 2. Navigation Architecture (Global Header & Sidebar)

- Header: Minimalist Top Navigation featuring:

  - App Logo: Minimalist abstract skeleton joint / gait icon with title "GaitVision AI | Clinical Assessment".

  - Active Patient Context Bar: "Patient ID: #PX-80492 | Age: 58 | Sex: M".

  - Actions: PDF Export button, Print Clinical Report button, User Profile (Dr. Specialist).

- Left Sidebar: Collapsible navigation with icons (Patients List, New Gait Scan, Analytical History, System Metrics, Settings).

---

### 3. Screen 1: Gait Video Upload & Patient Intake Workspace

- Main Content Area (2-Column Grid Layout):

  - Left Panel (Patient Details & Metadata Input):

    - Form fields for Patient ID, Age, Gender, Primary Clinical Notes, and Walk Test Conditions (e.g., Treadmill vs. Open Hallway, 10-meter Walk Test).

    - Select dropdown for primary observation concern (e.g., Post-Stroke, Parkinsonian Tremor, General Ataxia).

  - Right Panel (RGB Gait Video Dropzone):

    - Large drag-and-drop file upload zone supporting `.mp4`, `.avi`, and `.mov` RGB Gait Videos.

    - Embedded mini video preview player with play/pause, scrub bar, and resolution indicator ($T \times 224 \times 224$ downsampling indicator).

    - Modern processing toggle: "Include Joint-Level Heatmap Extraction" & "Generate Temporal Gait-Phase Breakdown".

    - Primary Call to Action Button: Prominent blue "Run Explainable Gait Analysis" button with a sleek, non-intrusive loading state indicator.

---

### 4. Screen 2: Diagnostic & Explainability Output Dashboard

(Structured in a 3-Column Bento Grid Layout for clear multi-modal analysis)

#### Column 1: Synchronized Video & Spatial Attention Viewer

- Primary Video Player Card:

  - Video player displaying input RGB video frame-by-frame.

  - Interactive overlay toggle buttons: "Raw RGB Video", "Sparse Skeletonization", "Spatial Attention Overlay".

  - Synchronized gait timeline scrubber showing current frame number $T$ and active Gait Phase badge (Heel Strike, Loading, Midstance, Terminal Stance, Swing Phase).

#### Column 2: Deep Explainability & Anatomic Localization

- Anatomic Heatmap Card:

  - Interactive 2D Body Joint Skeleton Diagram (Torso, Hip, Knee, Ankle, Foot).

  - Heatmap intensity scale (Blue to Red) indicating localized spatial abnormality contribution per joint region.

- Temporal Curve Graph Card:

  - Clean line chart showing temporal attention weights across video frames (X-axis: Frame Sequence $1 \dots T$, Y-axis: Anomaly Attention Score).

  - Highlighted color bands corresponding to gait phases (Heel Strike through Swing Phase).

#### Column 3: Softmax Diagnostic Classification & Clinical Report

- Abnormality Probabilities Card:

  - Softmax classification result card displaying likelihood percentage bars across the 6 target classes:

    1. Normal

    2. Hemiplegic

    3. Parkinsonian

    4. Ataxic

    5. Spastic

    6. Neuropathic

  - Highest confidence class highlighted with a clear diagnostic badge and confidence percentage score.

- Integrated Clinical Summary & Doctor Notes Panel:

  - Auto-generated narrative clinical summary: "Key findings indicate asymmetric temporal attention during the Midstance phase, localized predominantly in the right knee and ankle joint regions."

  - Editable rich-text box for physician remarks.

  - Action footer: Primary button to "Save to Patient Record" and secondary button to "Download PDF Diagnostic Report".

---

### 5. UI Micro-Interactions & Details

- Tooltips on clinical terminology explaining metric interpretations.

- Smooth transitions between raw video playback and attention map heatmaps.

- High contrast, accessible visual charts designed specifically for daylight clinical viewing environments.

This project was built with [Lovable](https://lovable.dev).

**Live app**: https://gaitvision-insight.lovable.app

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/805accc1-31dd-4772-ae6f-239219fbf052).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
