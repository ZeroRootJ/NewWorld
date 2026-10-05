# Elsevier-Style Research Figure Guidelines for Codex

Use this file as the default specification whenever creating, revising, or exporting research graphs and multi-panel figures for an Elsevier journal.

> **Priority rule:** The target journal's current *Guide for Authors* and any explicit user instructions override this file. If the journal is known, check its requirements before final export. Do not claim that a figure is submission-ready unless the journal-specific requirements have been checked.

## 1. Objective

Produce figures that remain clear, consistent, and legible at their **final published size**, not merely when enlarged on screen. Favor restrained, publication-ready graphics over presentation-style graphics.

## 2. Required Defaults

### 2.1 Final figure size

Choose one intended publication width before styling or exporting:

| Layout | Final width | Approximate width |
|---|---:|---:|
| Minimal element | 30 mm | 1.18 in |
| Single column | 90 mm | 3.54 in |
| 1.5 columns | 140 mm | 5.51 in |
| Double column / full width | 190 mm | 7.48 in |

- Default to **90 mm** for a simple single-panel plot.
- Use **190 mm** for dense plots, wide plots, or multi-panel figures that would be illegible at single-column width.
- Use **140 mm** only when the target journal supports an intermediate-width layout.
- Set the canvas to the intended final dimensions before judging text, markers, line widths, or spacing.
- Keep the figure height proportional to its content. Do not stretch the plot or add large empty margins.
- Crop tightly while preserving all labels and annotations.

### 2.2 Typography

- Use one font family consistently throughout all figures in the manuscript.
- Preferred Elsevier-supported fonts are **Arial or Helvetica**, **Times or Times New Roman**, **Courier**, and **Symbol**.
- Default plotting font: **Arial**. If Arial is unavailable, use Helvetica or a metrically compatible sans-serif font and embed it on export.
- At final printed size, use:
  - **7 pt** for tick labels, axis labels, legend text, annotations, and ordinary figure text.
  - **No smaller than 6 pt** for subscripts and superscripts.
  - **8–9 pt** for panel labels when needed for hierarchy.
- Use regular or medium weight for most text. Reserve bold for panel labels or a genuinely necessary emphasis.
- Use sentence case. Keep wording compact and terminology consistent across figures.
- Use proper mathematical notation, true minus signs, and consistent significant figures.
- Put units in parentheses after the quantity where appropriate, for example `Time (s)` and `Concentration (mg L−1)`.
- Embed fonts in vector files. If reliable embedding is impossible, convert text to outlines only in the final archival export and retain an editable source with live text.

### 2.3 Axes, lines, markers, and grids

- Recommended minimum line width: **0.25 pt** at final size. Never use a line below Elsevier's absolute minimum of **0.1 pt**.
- Use approximately **1.0 pt** for primary data curves or other prominent lines.
- Suggested hierarchy at final size:
  - primary data: **0.8–1.2 pt**;
  - secondary data, axes, and error bars: **0.5–0.8 pt**;
  - gridlines or reference guides: **0.25–0.5 pt**.
- Make markers distinguishable at final size; as a practical default, use **4–6 pt** markers and increase them when the plot is dense.
- Keep tick marks visible but unobtrusive. Avoid overly heavy boxes around the plotting area.
- Use gridlines only when they materially help value comparison. Keep them visually subordinate to the data.
- Avoid decorative borders, gradients, shadows, 3-D effects, and unnecessary background fills.
- Do not use line smoothing that changes the scientific meaning of the data.

### 2.4 Legends and labels

- Label axes with both the measured quantity and unit unless the quantity is dimensionless.
- Prefer direct labeling when it reduces eye movement and does not crowd the data.
- Otherwise, place the legend in unused space and ensure it does not cover data, uncertainty bands, or annotations.
- Keep legend order consistent with visual order or scientific importance.
- Do not duplicate information in a title, legend, annotation, and caption.
- Do not place the full figure caption inside the artwork. Supply it separately in the manuscript or submission system.
- Do not add `Figure 1` or `Fig. 1` inside the plot unless the target journal explicitly requests it.

### 2.5 Multi-panel figures

- Use uppercase panel labels: **A, B, C, ...**
- Place panel labels consistently, normally at the upper-left of each panel.
- Use identical font sizes, line widths, marker conventions, axis styling, and spacing across panels.
- Align plot areas, axes, and shared baselines precisely.
- Use shared axes and a shared legend when this improves clarity without creating ambiguity.
- Remove repeated labels only when the remaining labels make the shared meaning unmistakable.
- Keep gaps between panels small but sufficient to prevent labels from colliding.

## 3. Color Handling

- **Do not invent, replace, normalize, or reinterpret domain-specific color codes.** Preserve colors supplied by the user, source data, or domain convention.
- Do not apply a generic palette merely to make a figure look more polished.
- When permitted by the scientific convention, supplement color with line style, marker shape, hatching, position, or direct labels so the figure remains interpretable in grayscale and for readers with color-vision deficiency.
- Maintain adequate contrast between data, annotations, and background.
- Use RGB for color artwork unless the target journal explicitly asks for another color space.

## 4. Export and Resolution

Prefer a vector master for graphs, diagrams, and line-based plots. Keep editable source files and export each figure as a separate file.

| Artwork type | Preferred output | Minimum effective resolution at final size |
|---|---|---:|
| Vector graphs / diagrams | PDF or EPS with embedded fonts, subject to journal instructions | Resolution-independent |
| Pure bitmap line art | TIFF, bitmap mode | 1000 dpi; use 1200 dpi for very fine lines |
| Combination art: lines/text plus raster content | TIFF or maximum-quality JPEG | 500 dpi |
| Color or grayscale photographs / continuous-tone images | TIFF or maximum-quality JPEG | 300 dpi |

- Do not use screenshots for plots.
- Do not upscale a low-resolution image and treat the new DPI metadata as added detail.
- Calculate raster dimensions from the final physical size: `pixels = dpi × width in inches`.
- Useful minimum pixel widths from Elsevier's sizing guidance:

| Final width | 300 dpi | 500 dpi | 1000 dpi |
|---:|---:|---:|---:|
| 30 mm | 354 px | 591 px | 1181 px |
| 90 mm | 1063 px | 1772 px | 3543 px |
| 140 mm | 1654 px | 2756 px | 5512 px |
| 190 mm | 2244 px | 3740 px | 7480 px |

- Use lossless or maximum-quality export. Avoid repeated JPEG recompression.
- Ensure all linked or embedded raster elements meet their required effective resolution after scaling.
- Use a logical filename such as `Fig01_main_result.pdf`, and keep numbering consistent with the manuscript.

## 5. Scientific Integrity and Content Rules

- Never alter, omit, move, or emphasize data in a way that changes its interpretation.
- Global adjustments to brightness, contrast, or color balance may be made to an image only when they do not obscure or eliminate information.
- Show uncertainty, sample size, statistical definitions, normalization, and transformations where scientifically relevant.
- Define nonstandard abbreviations and symbols in the caption or legend.
- Include scale bars in images when scale matters. Do not rely only on a magnification statement.
- Retain the data-to-figure script, source data, and editable figure source whenever possible for reproducibility.

## 6. Recommended Codex Workflow

When asked to create a figure, Codex should:

1. Identify the target journal, intended column width, figure type, number of panels, and required output formats from the prompt or project context.
2. If the target journal is known and current requirements are accessible, check its *Guide for Authors*. Otherwise, use the defaults in this file and explicitly state that journal-specific compliance remains to be checked.
3. Preserve any domain-specific color mapping exactly as supplied.
4. Build the figure at its intended final physical dimensions.
5. Apply the typography, line, marker, layout, and labeling rules above.
6. Export an editable/vector version when possible and a high-resolution raster version only when needed.
7. Inspect the exported file at **100% final size**, not only in a zoomed plotting window.
8. Run the checklist below and report any unresolved compliance issue.

## 7. Final Quality-Control Checklist

Before delivery, verify all items:

- [ ] Target journal requirements were checked, or the lack of a journal-specific check is disclosed.
- [ ] The intended final width is documented: 30, 90, 140, or 190 mm, unless the journal specifies another size.
- [ ] Ordinary text is approximately 7 pt at final size; subscripts and superscripts are at least 6 pt.
- [ ] A supported, consistent font is used and embedded in vector output.
- [ ] Primary lines are clear; no line is below 0.1 pt; ordinary line work is at least 0.25 pt.
- [ ] Markers, error bars, annotations, and symbols remain distinct at final size.
- [ ] Axis labels include quantities and units where applicable.
- [ ] Legends and annotations do not obscure data.
- [ ] Multi-panel labels and alignments are consistent.
- [ ] Domain-specific colors were preserved and no arbitrary palette was introduced.
- [ ] The figure remains interpretable without relying on color alone where scientifically permissible.
- [ ] The canvas is tightly cropped with no clipped text or excessive whitespace.
- [ ] Raster elements meet 300, 500, or 1000 dpi as appropriate at final size.
- [ ] No low-resolution image was artificially upscaled.
- [ ] Fonts, transparent elements, symbols, and line styles render correctly in the exported file.
- [ ] The caption is separate from the artwork unless the journal explicitly requires otherwise.
- [ ] The source data, generation code, and editable master are retained.

## 8. Short Instruction Block for Reuse

Copy the following into a Codex request when a compact directive is useful:

```text
Create the figure according to ELSEVIER_FIGURE_GUIDELINES.md. Build it at the intended final publication width and make all typography and line weights valid at that size. Use Arial by default, 7 pt ordinary text, at least 6 pt subscripts/superscripts, approximately 1 pt primary data lines, and no line below 0.1 pt. Preserve all domain-specific color codes exactly; do not substitute a generic palette. Prefer vector output with embedded fonts for graphs. If raster output is required, use the appropriate effective resolution at final size: 1000 dpi for line art, 500 dpi for combination art, or 300 dpi for continuous-tone images. Check the exported result for clipping, crowding, legibility, and consistency. The target journal's current Guide for Authors overrides these defaults.
```

## 9. Sources

These defaults are based on Elsevier's general artwork guidance, accessed 5 October 2026:

- [Elsevier — Artwork and media instructions](https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions)
- [Elsevier — Artwork overview and supported fonts](https://www.elsevier.com/en-in/about/policies-and-standards/author/artwork-and-media-instructions/artwork-overview)
- [Elsevier — Artwork types and resolution requirements](https://www.elsevier.com/en-gb/about/policies-and-standards/author/artwork-and-media-instructions/artwork-types)
- [Elsevier — Artwork sizing](https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing)
- [Elsevier — Artwork FAQ and line-weight guidance](https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-faq)

Elsevier's general guidance is not a substitute for the target journal's current instructions.
