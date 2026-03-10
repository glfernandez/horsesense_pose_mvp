# Spec-Driven Development Framework (HorseSense)

## Purpose
Ensure every implementation step traces back to a written objective, spec, and acceptance criteria before code is added.

## Required Artifacts
- `NORTH_STAR.md`
- `OBJECTIVE.md`
- `PRODUCT_SPEC.md`
- `TECH_SPEC.md`
- `ROADMAP.md`
- `TASKS.md`
- `ACCEPTANCE_CRITERIA.md`
- `DECISIONS.md`
- `RISK_REGISTER.md`

## Workflow
1. Clarify objective and MVP boundaries
2. Write/confirm product and technical specs
3. Break work into tasks with explicit acceptance criteria
4. Implement smallest end-to-end vertical slice
5. Validate against acceptance criteria
6. Record decisions/deviations
7. Iterate next slice

## Change Control
If a change affects deliverables, interfaces, or acceptance thresholds:
- Update `PRODUCT_SPEC.md` and/or `TECH_SPEC.md`
- Add a decision note in `DECISIONS.md`
- Update `TASKS.md` and acceptance criteria as needed
