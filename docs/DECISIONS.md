# Decisions Log

## 2026-02-25 - Spec-driven scaffold created before DLC integration
- Decision: Build documentation, project structure, CLI skeleton, and feature/report modules first.
- Why: Reduces execution ambiguity for agents and keeps work aligned with explicit deliverables and acceptance criteria.
- Impact: Faster onboarding and easier handoff; DLC integration remains next implementation step.

## 2026-02-25 - `requirements.txt` chosen over `environment.yml`
- Decision: Start with `requirements.txt` for portability and quick setup.
- Why: Simpler initial bootstrap; target hardware/GPU specifics for torch can vary.
- Impact: May add `environment.yml` later for a pinned office-laptop environment.

## 2026-03-02 - Separate business-planning workspace kept outside `horsesense_pose_mvp`
- Decision: Store owner investment, revenue, and commercial analysis in a root-level `stable_business_analysis/` folder instead of mixing it into pose MVP docs.
- Why: The commercial planning material is useful context, but it is outside the current MVP deliverables and acceptance criteria.
- Impact: Pose MVP scope stays clean while related business analysis remains easy to review and expand.

## 2026-03-02 - Business analysis uses explicit assumption tracking before modeling
- Decision: Add a business developer interview guide, source-data checklist, and assumptions register before refining any commercial model.
- Why: The current investment roadmap contains many unvalidated assumptions that could distort owner-facing planning if treated as facts.
- Impact: Commercial planning can now separate confirmed inputs from estimates and build safer downside/base/upside cases.
