---
name: lunch
description: Pick restaurants for the next team lunch from README.md, always excluding places already marked visited. Use when the user asks for lunch suggestions, team lunch ideas, or wants to pick a restaurant.
---

# Lunch Picker

Suggest restaurants for the team's next lunch from this repo's `README.md`.

## Steps

1. Read `README.md` in the repo root.
2. Parse every restaurant entry under the location sections (skip the `## Activities` section unless the user explicitly asks for an activity instead of a restaurant).
   - `- [x] **[Name](link)** — address` → already visited. Never suggest these.
   - `- [ ] **[Name](link)** — address` → not yet visited. Eligible.
3. From the eligible (unchecked) list, pick 4 restaurants, unless the user specifies a different number.
   - Default to variety: don't cluster all 4 in one neighborhood or cuisine unless the user asks for that (e.g. "something downtown" or "sushi").
   - If the user gives constraints (location, cuisine, budget, walking distance from Lilly, etc.), filter to those first, then pick from what's left.
   - If fewer than the requested number of eligible spots remain, say so and suggest what's available.
4. Present the picks as a short numbered list: name, neighborhood/section, and a one-line reason pulled from its description in the README.
5. Offer to mark one as visited (flip `[ ]` to `[x]`) once the user tells you where they actually went — do this with an Edit to `README.md`, don't ask for confirmation beyond that.

## Notes

- "No repeats" is the default behavior, not an opt-in — always exclude `[x]` entries without being asked.
- Don't edit README.md just to generate suggestions; only edit it when the user confirms a place was visited.
- If the user asks to add a brand new restaurant to the list, follow the format and guidelines already documented in this repo's `CLAUDE.md`.
