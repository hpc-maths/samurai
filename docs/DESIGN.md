---
name: samurai documentation
description: An illustrated book on adaptive meshes, set in serif on white with one red, and plates drawn from code.
colors:
  paper: "#ffffff"
  margin-grey: "#f3f4f6"
  slate-ink: "#1e2832"
  secondary-ink: "#4f5b67"
  muted-ink: "#5f6a75"
  rule-grey: "#d5dae0"
  soft-rule: "#e7eaee"
  logo-red: "#af0000"
  red-wash: "#f8e4e4"
  red-selection: "#f3d0d0"
  figure-line: "#bfc7cf"
  ink-wash: "#e8ebef"
  literal-blue: "#2d5872"
  comment-grey: "#5c6773"
typography:
  display:
    fontFamily: "Source Serif 4, Georgia, serif"
    fontSize: "2.874rem"
    fontWeight: 400
    lineHeight: 1.12
    letterSpacing: "-0.012em"
  headline:
    fontFamily: "Source Sans 3, system-ui, sans-serif"
    fontSize: "0.8125rem"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "0.14em"
  title:
    fontFamily: "Source Serif 4, Georgia, serif"
    fontSize: "1.378rem"
    fontWeight: 600
    lineHeight: 1.3
  body:
    fontFamily: "Source Serif 4, Georgia, serif"
    fontSize: "1.1875rem"
    fontWeight: 400
    lineHeight: 1.62
  caption:
    fontFamily: "Source Serif 4, Georgia, serif"
    fontSize: "0.9975rem"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Source Sans 3, system-ui, sans-serif"
    fontSize: "0.72rem"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "0.13em"
  code:
    fontFamily: "Source Code Pro, ui-monospace, Menlo, monospace"
    fontSize: "0.879rem"
    fontWeight: 400
    lineHeight: 1.6
rounded:
  none: "0"
  sm: "2px"
  md: "3px"
spacing:
  frame-gap: "4px"
  plate-inner: "18px"
  plate-outset: "40px"
  legend-gutter: "40px"
  header: "4.75rem"
components:
  search-field:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.slate-ink}"
    rounded: "{rounded.md}"
    height: "2.25rem"
    width: "21rem"
    padding: "0 0.625rem 0 0.75rem"
  version-badge:
    textColor: "{colors.slate-ink}"
    rounded: "{rounded.md}"
    height: "2.125rem"
    padding: "0 0.625rem"
  section-head:
    textColor: "{colors.slate-ink}"
    typography: "{typography.headline}"
  table-header:
    textColor: "{colors.slate-ink}"
    typography: "{typography.label}"
    padding: "0.55em 1.1em 0.55em 0"
  statement-label:
    textColor: "{colors.logo-red}"
  plate-frame:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.slate-ink}"
    rounded: "{rounded.none}"
    padding: "{spacing.frame-gap}"
  plate-legend:
    textColor: "{colors.secondary-ink}"
    typography: "{typography.caption}"
    padding: "0 {spacing.legend-gutter}"
  listing:
    textColor: "{colors.slate-ink}"
    typography: "{typography.code}"
    padding: "0.7rem 0 0.75rem"
---

<!-- The typography role "title" in the frontmatter is not the page title. -->
<!-- markdownlint-configure-file {"MD025": {"front_matter_title": ""}} -->

# Design System: samurai documentation

## Overview

**Creative North Star: "The Illustrated Monograph"**
The documentation is set like a scientific book: a serif text column on white paper, numbered parts and chapters in the margin, definitions and remarks set as a textbook sets them, and plates that a reader studies the way they would study a figure in a treatise.

Every page is ink on paper with one red, taken from the logo. Color never fills an area for decoration; it marks the one thing a reader must find (the current chapter, a section head, a statement label, the cells a figure points out). Structure comes from rules, not boxes: thin rules above and below a listing, a heavy rule under the page title, a ruled double frame around a plate. All lines share three stroke weights, in the page and in the drawings, so text and figures read as one printed object.

The figures are drawn from code and true to the library: graded meshes, half-open intervals, the cells the program of the page prints. Their faces, strokes and colors are the page's own, so a plate follows the light and the dark themes like the text around it. Bold primary color fields with heavy black rules (a neoplastic look) are off-brand.

**Key Characteristics:**

- One serif for reading, one sans for labels in spaced capitals, one mono for code, plus a math face for operators.
- Ink in three tones, one red, no other hue outside syntax highlighting.
- Flat paper: no shadows, no tinted boxes, no rounded cards.
- Three stroke weights: hairline (0.5px), rule (1px), heavy rule (1.5px).
- Plates in a double frame with a centered legend, "Explanation of Plate N".
- Light and dark themes from one set of tokens, `--sm-*` in `docs/source/_static/css/samurai.css`.

## Colors

A cool slate ink on white, with a single deep red from the logo as the only accent.

Every color is a CSS variable on `body`, defined once for light and once for dark (both under `body[data-theme="dark"]` and under `prefers-color-scheme: dark` when no theme is forced). Furo's own variables are mapped onto these tokens in `docs/source/conf.py`. The frontmatter holds the light values; the dark values are listed here and in the sidecar.

### Primary

- **Logo Red** (`--sm-red`, #af0000; dark #f0736a): the current chapter's left stroke in the navigation and the current entry in "On this page", the short stroke of a section head, bullet markers, statement and admonition labels, link hover and underline, focus outlines, and in figures the result of an operation, the arrow of motion and the cells a figure points out.
- **Red Wash** (`--sm-red-wash`, #f8e4e4; dark #3f2023): the fill of red cells in figures, highlighted code lines and the target of an anchor.
- **Red Selection** (`--sm-red-select`, #f3d0d0; dark #5e2a2a): text selection only.

### Neutral

- **Paper** (`--sm-paper`, #ffffff; dark #131a21): the page, the header bar, the search field, and the halo behind figure text that crosses lines.
- **Margin Grey** (`--sm-side`, #f3f4f6; dark #171f28): the navigation column and the logo cell of the header.
- **Slate Ink** (`--sm-ink`, #1e2832; dark #d8dfe6): body text, titles, links, the heavy rule under a page title, table rules, plate frames, operand cells in figures.
- **Secondary Ink** (`--sm-ink-2`, #4f5b67; dark #a6b1bc): navigation links, header links, captions and legends, punctuation in code, axis labels in figures.
- **Muted Ink** (`--sm-ink-3`, #5f6a75; dark #8b97a3): chapter numbers, the rules of listings, statements and admonitions, the inner plate frame, file names under listings, notes in legends.
- **Rule Grey** (`--sm-rule`, #d5dae0; dark #2e3843): header and column borders, the long rule of a section head, input borders.
- **Soft Rule** (`--sm-rule-soft`, #e7eaee; dark #222b35): rules between table rows and the column rule of a plate legend.
- **Figure Line** (`--sm-fig-line`, #bfc7cf; dark #6e7b88): the faint outline of empty cells and the rule between operands and results in figures.
- **Ink Wash** (`--sm-ink-wash`, #e8ebef; dark #2e3a47): the fill of operand cells in figures.

### Syntax

- **Literal Blue** (`--sm-tok-str`, #2d5872; dark #93bfd8): strings and numbers in listings, the only hue besides red.
- **Comment Grey** (`--sm-tok-com`, #5c6773; dark #8d99a5): comments, in italics.

**The One Red Rule.** Red marks things and never fills an area. It appears as a stroke, a label, a marker or a hatched wash on the cells a figure is about, never as a background block or a button.

**The Token Only Rule.** No color is written as a literal in a page or a figure. A figure that needs a new color adds a `--sm-*` variable to `samurai.css`, in the light block and in both dark blocks. The one literal in the stylesheet is the light sheet (#f2efe8) placed behind raster images in dark mode, because those images are drawn on white.

## Typography

**Body Font:** Source Serif 4 (with Georgia, serif)
**Label Font:** Source Sans 3 (with system-ui, sans-serif)
**Mono Font:** Source Code Pro (with ui-monospace, Menlo, monospace)
**Math Face:** Noto Sans Math, declared as "samurai math" for the operator ranges only (U+2200 to U+2211, U+2213 to U+22FF, U+27E6 to U+27EF), placed first in each stack so ∇ and ∪ render, since Source Serif 4 lacks them.

**Character:** a book serif with optical sizing carries the reading; a quiet sans in spaced capitals labels the structure (parts, section heads, table headers, plate titles), the way running heads and captions label a printed page. All faces are self-hosted variable WOFF2 files in `docs/source/_static/fonts/`, with their licenses.

### Hierarchy

- **Display** (Source Serif 4 400, 2.42em of the body, line-height 1.12, tracking -0.012em): the page title, balanced, over a heavy ink rule.
- **Headline** (Source Sans 3 600, 0.684em of the body, uppercase, tracking 0.14em): the section head. It is followed on the same line by a 26px red stroke 2px tall, then a 1px rule to the end of the column. Section heads are kept small so that the text carries the page.
- **Title** (Source Serif 4 600, 1.16em of the body): subsections (h3); h4 is the same face at body size.
- **Body** (Source Serif 4 400, 1.1875rem, line-height 1.62; 1.0625rem below 46em): the text column. Links are ink with a 1px red underline offset 3px, red on hover.
- **Caption** (Source Serif 4 italic 400, 0.84em, secondary ink): figure and diagram captions. A numbered caption opens with its number in the label style.
- **Label** (Source Sans 3 600, 0.70 to 0.78rem, uppercase, tracking 0.10 to 0.16em): navigation parts, "On this page", previous and next, table headers, plate titles, legend titles.
- **Code** (Source Code Pro 400, 0.74em of the body, line-height 1.6, no ligatures, tab size 4): listings. Inline code is 0.84em with no box or tint; the face alone marks it.

**The Small Capitals Rule.** Statement labels ("Definition 1.", "Remark 2."), admonition titles, "Fig. N." in legends and in drawings are serif in all small capitals, tracking 0.06 to 0.07em, with kerning turned off so spaced capitals stay even. Definitions and their admonition titles are red; figure labels are ink.

**The Three Voices Rule.** Serif for what is read, sans for what labels, mono for what is typed. A label is never set in the serif and body text never in the sans.

## Layout

The page is Furo's three columns: the navigation on Margin Grey at the left, the text column, and "On this page" at the right.

- **Header** (4.75rem, sticky, in Furo's announcement bar, from `docs/source/_templates/page.html`): the logo in a cell exactly as wide as the navigation column (`calc(50% - 26em)`, at least 15em), then the search field, the Gallery and Hands-on course links, the version, GitHub and the theme toggle. Below 67em the header is hidden and Furo's mobile header takes over, with the logo mark beside the name.
- **Navigation:** top-level entries are parts, numbered in upper-case roman numerals in the label style; their children are chapters, numbered in arabic, in the serif at 0.9375rem. The current chapter is ink, weight 600, with a 1px red stroke on its left.
- **Rhythm:** section heads sit 3em above and 1em below; statements and admonitions 1.6 to 1.9em above; figures 1.8em; plates 2.4em above and 2.5em below.
- **Plates break the column:** a plate extends 40px into each margin on wide screens and fits the column below 46em.
- **Diagrams** are drawn 680 units wide (the text column) and plates 711 units wide (the inside of a plate frame); one drawing unit is one CSS pixel at natural size.

Breakpoints:

- **67em:** the site header gives way to Furo's mobile header.
- **46em:** the body drops to 1.0625rem; plates fit the column; a plate with panels shows its figures stacked, "half" panels two by two; a plate without panels, a "wide" panel and a diagram keep a legible width (640 or 560 units) and scroll sideways inside the frame, under a hint "Scroll sideways to see the whole figure →" in muted sans.

## Elevation & Depth

The system is flat. There are no shadows anywhere: Furo's admonition and table shadows are removed. Depth is conveyed by rules and by the paper of the navigation column against the paper of the text. The one layering effect is the paper halo behind figure text that crosses lines (a 4px paper stroke painted under the glyphs).

**The Printed Page Rule.** If it could not be printed with ink on paper, it does not belong: no shadow, no blur, no gradient fill, no glow.

## Shapes

Corners are square. Rules, frames, listings, tables, statements and plates have no radius. The only rounded corners are on the two header controls: the search field and the version badge (3px), and the keyboard hint inside the search field (2px).

Lines come in three weights, the same in the stylesheet and in the drawings (`HAIR`, `RULE`, `HEAVY` in `docs/source/_ext/samurai_figures/draw.py`):

- **Hairline** (0.5px): callout leaders, hatched result cells, horizontal fill rules.
- **Rule** (1px): every structural line (listing rules, statement rules, table row rules, the inner plate frame, column borders) and the default figure stroke.
- **Heavy rule** (1.5px): the rule under a page title, the top and bottom of a table, the outer plate frame, and in figures the outline that a figure compares against (a coarse cell, the operand's outline).

Figures add a fourth, fainter outline for empty cells (0.75px in Figure Line).

**The Closing Stroke Rule.** A block of text set apart (a definition, a remark, a note, a warning) opens with a full rule in Muted Ink and closes with a short 3.2em rule, like a theorem in a book. A block that ends on a listing closes with the listing's own rule instead.

## Components

### Search field

Quiet and bordered: paper background, 1px Rule Grey border, 3px radius, 21rem by 2.25rem, a 15px magnifier stroke icon, a placeholder in Secondary Ink, and a `/` key hint in mono. On focus the border darkens to Muted Ink; there is no glow.

### Version badge and header links

The version is a sans 600 label in a 1px bordered box (3px radius). Header links are sans 0.875rem in Secondary Ink, ink on hover, with no underline. Icon buttons (GitHub, theme toggle) are 2rem squares with 18px icons, Secondary Ink, ink on hover.

### Navigation

Parts in spaced capitals with roman numerals in Muted Ink; chapters in serif with arabic numbers in a fixed 1.4rem gutter; links in Secondary Ink, ink on hover, no hover background. The current chapter carries a 1px red left stroke, and "On this page" marks the current section with a 1px red stroke against its 1px left rule.

### Listings

A listing has no box. It sits between two 1px rules in Muted Ink, set in the mono at 0.74em. Highlighting is ink: keywords in ink weight 600, names in ink, punctuation and preprocessor lines in Secondary Ink, strings and numbers in Literal Blue, comments in Comment Grey italics, highlighted lines on Red Wash. A captioned listing names its file under the bottom rule, right-aligned, in the mono at 0.68em in Muted Ink. Program output may wrap instead of scrolling (`wrap-output`).

### Definitions and remarks

Set from the `definition` and `remark` directives (`docs/source/_ext/statements.py`) and numbered in one sequence per page. The label ("Definition 1.") is red serif small capitals, weight 600 (500 for a remark); the defined term follows in italics between upright parentheses. Opening and closing rules follow the Closing Stroke Rule.

### Admonitions

Notes, warnings and topics are set as remarks: no icon, no background, no border box, no shadow. The title is the admonition name in red serif small capitals at body size; the body follows in the text style.

### Tables

A book table: a heavy ink rule above and below, a 1px ink rule under the header row, Soft Rule between rows, no vertical rules, no fills. Header cells are in the label style; body cells in the serif at 0.87em with lining tabular figures. Cells pad on the right only, so the first column aligns with the text.

### Plate (signature component)

The illustration of a page, from the `plate` directive (`docs/source/_ext/plates.py`):

- **Frame:** a 1.5px ink outer frame, a 4px paper gap, a 1px Muted Ink inner frame. Inside, 18px of padding (10px below 46em).
- **Head:** the title in sans spaced capitals (0.605em, tracking 0.16em) at the left, "Plate N" in serif italic at the right, on a common baseline above a 1px Muted Ink rule.
- **Drawing:** the SVG of the figure function, full width of the frame. Each figure inside carries "Fig. N." in serif small capitals under it, optionally followed by an italic qualifier ("Fig. 1. t = 0").
- **Legend:** outside and under the frame, inset 40px. A centered "Explanation of Plate N" in the label style, then two columns (one with `:columns: 1`, at most 38em and centered) separated by a 1px Soft Rule. Each paragraph opens with "Fig. N." in ink small capitals; a single italic letter is a callout and is set slightly larger in ink; a paragraph entirely in italics is a note, in Muted Ink; a word in red names the red of the drawing.
- **Numbering and links:** plates are numbered in page order; `{ref}` on a labeled plate reads "Plate 1 (title)".
- **Narrow screens:** see Layout. The stacked panels reuse the one drawing through `<use>` with their own `viewBox`, and are hidden from screen readers, which read the drawing's own description.

### Diagram

An inline figure from the `diagram` directive: the drawing at column width with no frame, then an italic caption in Secondary Ink. Diagrams are not numbered; a label is linked with explicit text. Rows of cells are labeled on the left: operands by an italic name or a sans level label, results by their red mono expression. A level 0 number line runs along the bottom.

### Figure vocabulary

The primitives of `draw.py`, shared by every plate and diagram:

- **Faces:** labels and axis numbers in Source Sans 3 (`sm-fig-label`, 11 to 12 units); variable and qualifier names in serif italic (`sm-fig-math`); code and interval labels in the mono (`sm-fig-code`, 10.5 to 11.5 units); "Fig. N." in serif small capitals (`sm-fig-sc`, 14 units).
- **Cells:** a cell in the set is filled with Ink Wash and outlined in ink (an operand) or filled with Red Wash and outlined in red (a result); a cell outside the set is a faint 0.75px Figure Line outline.
- **Half-open brackets:** under a run of cells, a 1px line from a filled dot (radius 3) at the start to an open, paper-filled dot at the end, with its interval `[a, b)` in the mono on a paper ground below.
- **Hatching:** the cells a figure points out are Red Wash crossed by fine red hatching at 45 degrees (0.8px lines every 3.4 units). A region where the solution is constant is ruled with fine horizontal Muted Ink hairlines every 3 units.
- **Callouts:** a dot of radius 1.8 on the target (red when the target is red), a hairline leader, and an italic serif letter (15 units) with a paper halo. The legend explains each letter.
- **Arrows:** open arrow heads, in Secondary Ink for flow and in red for motion or for a result.
- **Guides:** dashed lines (3 3 for ghost cells, 2 3 in red for edges carried across rows); a dotted circle for a former position.
- **Meshes:** graded quadtrees from `build_mesh`, checked by `check_graded`, so a plate never shows a mesh samurai could not build.

## Do's and Don'ts

### Do

- **Do** take every color in a page or a figure from a `--sm-*` variable, and add a new variable in the light block and both dark blocks when one is missing.
- **Do** draw with the three stroke weights (0.5px, 1px, 1.5px), plus 0.75px Figure Line for empty cells.
- **Do** draw every interval as a half-open bracket with a filled start and an open end, labeled `[a, b)` in the mono.
- **Do** keep red for what the figure or the page is about: a result, a current position, a statement label.
- **Do** label structure in Source Sans 3 spaced capitals, and set "Fig. N.", "Definition N." and admonition titles in serif small capitals.
- **Do** give every plate its panels so it stacks on narrow screens, and say in the legend when the drawing uses fewer levels than the program.
- **Do** describe every drawing in its `label` for screen readers.

### Don't

- **Don't** write a hex or rgb value in a figure function or a page.
- **Don't** add shadows, rounded cards, tinted boxes or gradients; set things apart with rules.
- **Don't** add a second accent hue; Literal Blue stays inside code listings.
- **Don't** use admonition icons or colored admonition backgrounds.
- **Don't** draw a figure of program data by hand; write a figure function. Hand-written SVG (`:svg:`) is only for schematics that depend on no program data.
- **Don't** draw a mesh that is not graded, or an interval with two filled ends.
- **Don't** use bold primary color fields with heavy black rules, or Japanese folklore motifs (brush strokes, ink washes, katanas).
