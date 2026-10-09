"""The figures of ``LBM/test_cases.md``: the velocity sets and the setups of the test cases.

The figures read the programs of ``demos/LBM`` that the page names: the velocity
lists and the lattice velocities lambda of the ``new_*.cpp`` programs, the four
quadrant states of the Lax-Liu configurations, and the initial datum and the wall
conditions of the implosion box. A program that no longer matches what a figure
expects raises ``ValueError``, which fails the docs build.
"""

import math
import re
from fractions import Fraction
from pathlib import Path

from ..draw import (
    EMPTY,
    FIG,
    HAIR,
    HEAVY,
    INK,
    INK2,
    INK3,
    RED,
    WASH,
    circle,
    line,
    patterns,
    rect,
    text,
)
from ..registry import DIAGRAM_WIDTH, Drawing, figure

DEMOS = Path(__file__).resolve().parents[5] / "demos" / "LBM"

# A list of velocities in the C++ source, {{1, 0}, {0, 1}, ...} or {{0}, {1}, ...}.
_VECTOR = r"\{\s*-?\d+(?:\s*,\s*-?\d+)?\s*\}"
_VELOCITIES = re.compile(rf"\{{\s*({_VECTOR}(?:\s*,\s*{_VECTOR})*)\s*\}}")


def _source(name):
    """Return the text of the program ``name`` of ``demos/LBM``."""
    return (DEMOS / name).read_text(encoding="utf-8")


def _after(source, anchor, name):
    """Return the text of ``source`` after ``anchor``, raise if the program has no anchor."""
    at = source.find(anchor)
    if at < 0:
        raise ValueError(f'{name}: "{anchor}" not found, the figure no longer matches the program')
    return source[at + len(anchor) :]


def velocities(name, anchor):
    """Return the velocities that follow ``anchor`` in the program ``name``, in their order.

    Each velocity is a tuple of integers, of length 1 in 1D and 2 in 2D.
    """
    match = _VELOCITIES.search(_after(_source(name), anchor, name))
    if not match:
        raise ValueError(f"{name}: no velocity list after {anchor}")
    return [
        tuple(int(v) for v in vector.split(","))
        for vector in re.findall(r"\{([^{}]*)\}", match.group(1))
    ]


# A rule of the form ``if (flag) { lambda = std::max(lambda, 2.); }``, which raises
# lambda when a command line flag is set.
_RAISE = re.compile(r"if \((\w+)\)\s*\{\s*lambda\s*=\s*std::max\(lambda,\s*([\d.]+)\);[^\n]*\s*\}")


def lattice_velocity(name, flag=None):
    """Return the lattice velocity lambda that the program ``name`` runs with.

    Without ``flag``, the default value of ``lambda``; with ``flag``, the value
    once the command line flag ``flag`` is set. Any other assignment of
    ``lambda`` in the program raises ``ValueError``: the figure would not know it.
    """
    source = _source(name)
    match = re.search(r"double lambda\s*=\s*([\d.]+)", source)
    if not match:
        raise ValueError(f"{name}: no default lambda")
    rules = {rule.group(1): float(rule.group(2)) for rule in _RAISE.finditer(source)}
    if len(re.findall(r"^\s*lambda\s*=", source, re.MULTILINE)) != len(rules):
        raise ValueError(f"{name}: lambda is set by a rule the figure does not know")
    if flag is not None and flag not in rules:
        raise ValueError(f"{name}: no rule sets lambda under --{flag}")
    default = float(match.group(1))
    return max(default, rules[flag]) if flag is not None else default


# The velocity sets of the page. Each one is read from the program that the page
# names for it; ``uses`` are the programs that run the set, with the case they
# implement on the page and, when the case needs one, the flag that selects it.
VELOCITY_SETS = (
    (
        "D1Q2",
        ("new_D1Q2_advection.cpp", "velocity_scheme<dim, 2>("),
        (
            ("new_D1Q2_advection.cpp", "velocity_scheme<dim, 2>(", "advection"),
            # --burgers raises lambda, so that it stays above |u|
            ("new_D1Q2_advection.cpp", "velocity_scheme<dim, 2>(", "Burgers", "burgers"),
            ("new_D1Q222_euler_sod.cpp", "std::array<std::array<int, dim>, 2> vel", "Sod"),
        ),
    ),
    (
        "D1Q3",
        ("new_D1Q3_shallow_waters_dam.cpp", "velocity_scheme<dim, 3>("),
        (("new_D1Q3_shallow_waters_dam.cpp", "velocity_scheme<dim, 3>(", "dam break"),),
    ),
    (
        "D1Q5",
        ("new_D1Q5_shallow_waters_dam.cpp", "velocity_scheme<dim, 5>("),
        (("new_D1Q5_shallow_waters_dam.cpp", "velocity_scheme<dim, 5>(", "dam break"),),
    ),
    (
        "D2Q4",
        ("new_D2Q4444_euler_lax_liu.cpp", "std::array<std::array<int, dim>, 4> vel"),
        (
            ("new_D2Q4444_euler_lax_liu.cpp", "std::array<std::array<int, dim>, 4> vel", "Lax-Liu"),
            (
                "new_D2Q5444_euler_rayleigh_taylor.cpp",
                "std::array<std::array<int, dim>, 4> vel4",
                "Rayleigh-Taylor",
            ),
            (
                "new_D2Q4444_euler_implosion.cpp",
                "std::array<std::array<int, dim>, 4> vel",
                "implosion",
            ),
        ),
    ),
    (
        "D2Q5",
        ("new_D2Q5444_euler_rayleigh_taylor.cpp", "std::array<std::array<int, dim>, 5> vel5"),
        (
            (
                "new_D2Q5444_euler_rayleigh_taylor.cpp",
                "std::array<std::array<int, dim>, 5> vel5",
                "Rayleigh-Taylor",
            ),
        ),
    ),
    (
        "D2Q9",
        ("new_D2Q9_von_karman.cpp", "velocity_scheme<dim, 9>("),
        (("new_D2Q9_von_karman.cpp", "velocity_scheme<dim, 9>(", "von Kármán"),),
    ),
)


def velocity_set(name, source, uses):
    """Return the velocities of the set ``name`` and its lambdas, ``[(lambda, [cases])]``.

    Every program of ``uses`` must list the same velocities, in the same order, as
    ``source``; the dimension and the number of velocities must match the name.
    """
    vel = velocities(*source)
    dim, q = int(name[1]), int(name[3:])
    if len(vel) != q or any(len(v) != dim for v in vel):
        raise ValueError(f"{source[0]}: {name} expects {q} velocities in {dim}D, got {vel}")
    lambdas = {}
    for program, anchor, case, *flag in uses:
        if velocities(program, anchor) != vel:
            raise ValueError(f"{program}: the {name} velocities differ from {source[0]}")
        lambdas.setdefault(lattice_velocity(program, *flag), []).append(case)
    return vel, sorted(lambdas.items())


def _lam(value):
    """Return the lambda markup ``λ = value``."""
    return f'<tspan class="sm-fig-math">λ</tspan> = {value:g}'


def _c(k, size):
    """Return the label c_k of a velocity: an italic c, an upright subscript k."""
    return (
        f'c<tspan dy="{0.25 * size:g}" font-size="{0.72 * size:g}"'
        f' font-style="normal">{k}</tspan>'
    )


def _arrow(p, x1, y1, x2, y2):
    """Return a velocity arrow from (x1, y1) to (x2, y2)."""
    return line(x1, y1, x2, y2, stroke=INK2, extra=f'marker-end="url(#{p}-a)"')


SPAN = 44  # one lattice step, in drawing units
LABEL = 14  # the size of a velocity label
REST = 3.2  # the radius of the rest velocity dot


def _rest(cx, cy):
    """Return the dot of the rest velocity, the key under the set names it."""
    return circle(cx, cy, REST, fill=INK)


KEY = 12  # the size of the key text
KEY_LINE = 19  # from the key line to the first lambda line
# the width of the key, its dot and its text, in 1D and 2D, measured in the browser
KEY_WIDTH = {1: 77.1, 2: 95.6}


def _key(cx, y, k, dim):
    """Write the key of the rest velocity c_k, centered on cx: its dot, then its name."""
    x = cx - KEY_WIDTH[dim] / 2
    zero = "0" if dim == 1 else "(0, 0)"
    low = 0.25 * KEY
    name = (
        f'<tspan class="sm-fig-math">c</tspan><tspan class="sm-fig-math" dy="{low:g}"'
        f' font-size="{0.72 * KEY:g}" style="font-style: normal">'
        f'{k}</tspan><tspan dy="{-low:g}"> = {zero}, at rest</tspan>'
    )
    return circle(x + REST, y - 0.35 * KEY, REST, fill=INK) + text(
        x + 2 * REST + 6, y, name, size=KEY, fill=INK
    )


def _nodes_1d(cx, cy):
    """Return the lattice nodes -2 to 2 on a line, with their coordinates under them."""
    g = line(cx - 2 * SPAN, cy, cx + 2 * SPAN, cy, stroke=FIG, sw=EMPTY)
    for k in range(-2, 3):
        g += circle(cx + k * SPAN, cy, 2.2, fill=FIG)
        g += text(
            cx + k * SPAN, cy + 44, f"{k}", anchor="middle", size=10.5, fill=INK3, cls="sm-fig-code"
        )
    return g


def _set_1d(p, cx, cy, vel):
    """Draw a 1D velocity set: the unit velocities on the line, the longer ones below it.

    The velocities of length 2 leave the center node on a second track, 20 units
    below, joined to the node by a hairline, so that no arrow covers another.
    """
    g = _nodes_1d(cx, cy)
    if any(abs(v[0]) == 2 for v in vel):
        g += line(cx, cy, cx, cy + 20, stroke=INK3, sw=HAIR)
    for k, (c,) in enumerate(vel):
        if c == 0:
            g += _rest(cx, cy)
            continue
        s = 1 if c > 0 else -1
        tip = cx + c * SPAN - s * 4
        if abs(c) == 1:
            # above the node it reaches, clear of the line of nodes
            g += _arrow(p, cx + s * (REST + 1), cy, tip, cy)
            g += text(
                tip + s * 4,
                cy - 14,
                _c(k, LABEL),
                anchor="middle",
                size=LABEL,
                fill=INK,
                cls="sm-fig-math",
            )
        else:
            g += _arrow(p, cx, cy + 20, tip, cy + 20)
            g += text(
                tip + s * 8,
                cy + 24,
                _c(k, LABEL),
                anchor="start" if s > 0 else "end",
                size=LABEL,
                fill=INK,
                cls="sm-fig-math",
            )
    return g


def _set_2d(p, cx, cy, vel):
    """Draw a 2D velocity set on its 3 by 3 lattice nodes, y pointing up."""
    g = ""
    for k in (-1, 0, 1):
        g += line(cx - SPAN, cy + k * SPAN, cx + SPAN, cy + k * SPAN, stroke=FIG, sw=EMPTY)
        g += line(cx + k * SPAN, cy - SPAN, cx + k * SPAN, cy + SPAN, stroke=FIG, sw=EMPTY)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            g += circle(cx + i * SPAN, cy + j * SPAN, 2.2, fill=FIG)
    for k, (vx, vy) in enumerate(vel):
        if (vx, vy) == (0, 0):
            g += _rest(cx, cy)
            continue
        norm = math.hypot(vx, vy)
        ux, uy = vx / norm, -vy / norm  # the drawing's y points down
        nx, ny = cx + vx * SPAN, cy - vy * SPAN
        g += _arrow(p, cx + ux * (REST + 1), cy + uy * (REST + 1), nx - ux * 4, ny - uy * 4)
        anchor = "middle" if vx == 0 else ("start" if vx > 0 else "end")
        lx = nx + 8 * (vx > 0) - 8 * (vx < 0)
        if vy == 0:
            ly = ny + 5
        elif vx == 0:
            ly = ny - 9 if vy > 0 else ny + 19
        else:
            ly = ny - 5 if vy > 0 else ny + 15
        g += text(lx, ly, _c(k, LABEL), anchor=anchor, size=LABEL, fill=INK, cls="sm-fig-math")
    return g


def _uses(cx, y, lambdas):
    """Write one line per lambda under a set: the value and the cases that use it."""
    g = ""
    for r, (value, cases) in enumerate(lambdas):
        g += text(
            cx,
            y + r * 17,
            f'{_lam(value)}  <tspan fill="{INK2}">{", ".join(cases)}</tspan>',
            anchor="middle",
            size=12,
            fill=INK,
        )
    return g


def _describe(name, vel, lambdas):
    """Describe a velocity set for screen readers."""
    vectors = ", ".join(f"c{k} = ({', '.join(str(v) for v in c)})" for k, c in enumerate(vel))
    uses = "; ".join(f"lambda = {value:g} for {', '.join(cases)}" for value, cases in lambdas)
    return f"{name}: {vectors}; {uses}."


@figure
def lbm_velocity_sets(p):
    """Draw the six velocity sets of the page, in the order of the programs.

    Figure of LBM/test_cases.md. Each set is drawn on its lattice nodes, the
    velocity c_k as an arrow from the center node to the node it reaches in one
    time step, the rest velocity as the center dot; under each set, a key line that
    names the rest velocity (empty for a set without one), then its lattice
    velocities lambda and the cases that use them.
    """
    sets = [(name, *velocity_set(name, source, uses)) for name, source, uses in VELOCITY_SETS]
    pitch = DIAGRAM_WIDTH / 3
    g, labels = "", []
    # the title, the center node and the key line of each row; the lambdas follow the key
    rows = ((16, 64, _set_1d, 134), (226, 318, _set_2d, 412))
    for r, (title_y, cy, draw_set, uses_y) in enumerate(rows):
        for col, (name, vel, lambdas) in enumerate(sets[3 * r : 3 * r + 3]):
            cx = (col + 0.5) * pitch
            g += text(cx, title_y, name, anchor="middle", size=13, fill=INK, cls="sm-fig-label")
            g += draw_set(p, cx, cy, vel)
            rest = [k for k, c in enumerate(vel) if not any(c)]
            if rest:
                g += _key(cx, uses_y, rest[0], len(vel[0]))
            g += _uses(cx, uses_y + KEY_LINE, lambdas)
            labels.append(_describe(name, vel, lambdas))
    return Drawing(
        DIAGRAM_WIDTH,
        rows[-1][3] + KEY_LINE + 2 * 17 + 4,
        "Six velocity sets on their lattice nodes, each velocity c_k an arrow from the center"
        " node, the rest velocity c_0 the center dot, named in a key under its set. "
        + " ".join(labels),
        patterns(p),
        g,
    )


def _numbers(block):
    """Return the numbers of a C++ initializer ``{a, b, c, d}`` as floats."""
    return [float(v) for v in block.split(",")]


# The quadrants as new_D2Q4444_euler_lax_liu.cpp orders its states, (x side, y side).
QUADRANTS = (("L", "L"), ("L", "U"), ("R", "L"), ("R", "U"))
_QUADRANT_OF = "const std::size_t q = (x < 0.5) ? (y < 0.5 ? 0 : 1) : (y < 0.5 ? 2 : 3);"


def lax_liu_states(config):
    """Return the quadrant states of the Lax-Liu ``config`` and its final time.

    The states come from ``new_D2Q4444_euler_lax_liu.cpp``: a dictionary from the
    quadrant name (``"UR"``, ``"UL"``, ``"LL"``, ``"LR"``) to ``(rho, u, v, p)``.
    """
    name = "new_D2Q4444_euler_lax_liu.cpp"
    source = _source(name)
    if _QUADRANT_OF not in source:
        raise ValueError(f"{name}: the quadrants are no longer split at x = 0.5 and y = 0.5")
    case = _after(source, f"case {config}:", name)
    case = case[: case.find("break;")]
    fields = {}
    for key in ("rho0", "ux0", "uy0", "p0"):
        match = re.search(rf"{key}\s*=\s*\{{([^}}]*)\}}", case)
        if not match:
            raise ValueError(f"{name}: no {key} in configuration {config}")
        fields[key] = _numbers(match.group(1))
    final = re.search(r"Tf_cfg\s*=\s*([\d.]+)", case)
    states = {
        y + x: tuple(fields[key][k] for key in ("rho0", "ux0", "uy0", "p0"))
        for k, (x, y) in enumerate(QUADRANTS)
    }
    return states, float(final.group(1))


SPEED = 30  # drawing units per unit of velocity


def _quadrant(p, x, y, w, name, state):
    """Draw the state of one quadrant: its name and values at the top, its velocity in red."""
    rho, u, v, pressure = state
    g = text(x + w - 9, y + 20, name, anchor="end", size=11, fill=INK3, cls="sm-fig-label")
    rows = (("ρ", rho), ("u", u), ("v", v), ("p", pressure))
    for r, (symbol, value) in enumerate(rows):
        g += text(
            x + 10,
            y + 20 + r * 15,
            f'<tspan class="sm-fig-math">{symbol}</tspan> = {value:g}',
            size=11.5,
            fill=INK,
        )
    if u or v:
        ox, oy = x + 0.55 * w, y + 0.86 * w
        g += circle(ox, oy, 1.8, fill=RED)
        g += line(
            ox, oy, ox + u * SPEED, oy - v * SPEED, stroke=RED, extra=f'marker-end="url(#{p}-ar)"'
        )
    return g


def _unit_square(x, y, w, tick=4):
    """Draw the frame of [0, 1]^2 with the ticks 0, 1/2 and 1 on both axes.

    The ticks are ``tick`` units long, out of the frame, on the bottom and left sides.
    """
    g = rect(x, y, w, w, stroke=INK, sw=HEAVY)
    for t, label in ((0, "0"), (0.5, "1/2"), (1, "1")):
        g += line(x + t * w, y + w, x + t * w, y + w + tick, stroke=INK)
        g += text(x + t * w, y + w + tick + 13, label, anchor="middle", size=11, cls="sm-fig-code")
        g += line(x - tick, y + w - t * w, x, y + w - t * w, stroke=INK)
        g += text(x - tick - 4, y + w - t * w + 4, label, anchor="end", size=11, cls="sm-fig-code")
    return g


@figure
def lax_liu_configurations(p):
    """Draw the four quadrant states of the Lax-Liu configurations 3 and 12.

    Figure of LBM/test_cases.md, from ``new_D2Q4444_euler_lax_liu.cpp``: the unit
    square split at x = 1/2 and y = 1/2, the values of rho, u, v and p in each
    quadrant, and the velocity (u, v) as a red arrow, 30 units per unit of velocity.
    """
    w, top = 250, 30
    g, labels = "", []
    for k, config in enumerate((3, 12)):
        states, final = lax_liu_states(config)
        x = 62 + k * (w + 92)
        g += text(
            x + w / 2,
            top - 12,
            f'Configuration {config}, <tspan class="sm-fig-math">T</tspan> = {final:g}',
            anchor="middle",
            size=12.5,
            fill=INK,
        )
        g += line(x + w / 2, top, x + w / 2, top + w, stroke=INK)
        g += line(x, top + w / 2, x + w, top + w / 2, stroke=INK)
        g += _unit_square(x, top, w)
        for name, (sx, sy) in (("UL", (0, 0)), ("UR", (1, 0)), ("LL", (0, 1)), ("LR", (1, 1))):
            g += _quadrant(p, x + sx * w / 2, top + sy * w / 2, w / 2, name, states[name])
        labels.append(
            f"Configuration {config}, T = {final:g}: "
            + "; ".join(
                f"{name} rho = {s[0]:g}, u = {s[1]:g}, v = {s[2]:g}, p = {s[3]:g}"
                for name, s in sorted(states.items(), key=lambda item: "UL UR LL LR".find(item[0]))
            )
            + "."
        )
    return Drawing(
        DIAGRAM_WIDTH,
        top + w + 26,
        "Two unit squares, each split at x = 1/2 and y = 1/2 into four quadrants UL, UR, LL and"
        " LR, with the values of rho, u, v and p written in each quadrant and the velocity (u, v)"
        " drawn as a red arrow where it is not zero. " + " ".join(labels),
        patterns(p),
        g,
    )


# The conserved variables of the four D2Q4 blocks, in the order of the program.
BLOCKS = ("ρ", "ρu", "ρv", "E")
BLOCK_NAMES = ("rho", "rho u", "rho v", "E")


def implosion_setup():
    """Return the setup of ``new_D2Q4444_euler_implosion.cpp``.

    A tuple ``(corner, (rho_in, rho_out), (p_in, p_out), odd_axis)``: the low
    state fills x + y <= ``corner``, and ``odd_axis[b]`` is the axis of the walls
    on which block b takes the anti-bounce-back condition, -1 for none.
    """
    name = "new_D2Q4444_euler_implosion.cpp"
    source = _source(name)
    found = [
        re.search(pattern, source)
        for pattern in (
            r"inside\s*=\s*\(x \+ y\) <= ([\d.]+);",
            r"rho\s*=\s*inside \? ([\d.]+) : ([\d.]+);",
            r"press\s*=\s*inside \? ([\d.]+) : ([\d.]+);",
            r"block_odd_axis\{([^}]*)\}",
            r"make_bc<samurai::BounceBack>\(f, velocities, block_sizes, block_odd_axis\)",
        )
    ]
    if not all(found):
        raise ValueError(f"{name}: the initial datum or the walls no longer match the figure")
    odd_axis = [int(v) for v in found[3].group(1).split(",")]
    if len(odd_axis) != len(BLOCKS):
        raise ValueError(f"{name}: expected {len(BLOCKS)} blocks, got {odd_axis}")
    return (
        float(found[0].group(1)),
        tuple(float(v) for v in found[1].groups()),
        tuple(float(v) for v in found[2].groups()),
        odd_axis,
    )


def _fraction(v):
    """Write ``v`` as the page does, a fraction such as 1/8 or 7/50, or an integer."""
    return str(Fraction(v).limit_denominator(100))


def _wall_hatch(x1, y1, x2, y2, nx, ny):
    """Hatch the outside of the wall from (x1, y1) to (x2, y2), ``(nx, ny)`` its outer normal."""
    g, n = "", int(math.hypot(x2 - x1, y2 - y1) // 8)
    for k in range(n + 1):
        t = k / n
        x, y = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        # short strokes at 45 degrees, leaning the same way on every wall
        g += line(x, y, x + 6 * nx - 6 * ny, y + 6 * ny + 6 * nx, stroke=INK3, sw=HAIR)
    return g


def _wall_text(x, y, axis, odd_axis, anchor):
    """Write the conditions of the walls normal to ``axis``, in two lines from (x, y)."""
    anti = [b for b, a in zip(BLOCKS, odd_axis) if a == axis]
    bounce = [b for b, a in zip(BLOCKS, odd_axis) if a != axis]

    def names(blocks):
        return ", ".join(f'<tspan class="sm-fig-math">{b}</tspan>' for b in blocks)

    return text(x, y, f"anti-bounce-back: {names(anti)}", anchor=anchor, size=12, fill=INK) + text(
        x, y + 17, f"bounce-back: {names(bounce)}", anchor=anchor, size=12, fill=INK2
    )


@figure
def implosion_box(p):
    """Draw the implosion box: the low state in the corner and the condition of each wall.

    Figure of LBM/test_cases.md, from ``new_D2Q4444_euler_implosion.cpp``: the unit
    square with the triangle x + y <= 1/2 in ink wash, the two states at rest, and on
    each side the variables that take the anti-bounce-back and the bounce-back
    conditions on that wall.
    """
    corner, (rho_in, rho_out), (p_in, p_out), odd_axis = implosion_setup()
    w, x, top = 220, 230, 60
    bottom = top + w
    leg = corner * w
    g = (
        f'<path d="M{x},{bottom} L{x + leg},{bottom} L{x},{bottom - leg} Z" fill="{WASH}"'
        f' stroke="{INK}" stroke-width="1"/>'
    )
    # the ticks reach past the hatching of the walls
    g += _unit_square(x, top, w, tick=11)
    for x1, y1, x2, y2, nx, ny in (
        (x, top, x + w, top, 0, -1),
        (x, bottom, x + w, bottom, 0, 1),
        (x, top, x, bottom, -1, 0),
        (x + w, top, x + w, bottom, 1, 0),
    ):
        g += _wall_hatch(x1, y1, x2, y2, nx, ny)
    # the states, at rest everywhere
    corner_s, rho_in_s, p_in_s = _fraction(corner), _fraction(rho_in), _fraction(p_in)
    rho_out_s, p_out_s = _fraction(rho_out), _fraction(p_out)
    rho, pressure = '<tspan class="sm-fig-math">ρ</tspan>', '<tspan class="sm-fig-math">p</tspan>'
    g += text(x + 12, bottom - 26, f"{rho} = {rho_in_s}", size=11.5)
    g += text(x + 12, bottom - 11, f"{pressure} = {p_in_s}", size=11.5)
    g += text(x + w - 14, top + 26, f"{rho} = {rho_out_s}", anchor="end")
    g += text(x + w - 14, top + 41, f"{pressure} = {p_out_s}", anchor="end")
    edge = '<tspan class="sm-fig-math">x</tspan> + <tspan class="sm-fig-math">y</tspan>'
    g += text(x + leg / 2 + 10, bottom - leg / 2 - 8, f"{edge} = {corner_s}", size=11.5, fill=INK)
    # the walls normal to x (left and right) and to y (top and bottom)
    g += _wall_text(x - 54, top + w / 2 - 4, 0, odd_axis, "end")
    g += _wall_text(x + w + 18, top + w / 2 - 4, 0, odd_axis, "start")
    g += _wall_text(x + w / 2, top - 34, 1, odd_axis, "middle")
    g += _wall_text(x + w / 2, bottom + 50, 1, odd_axis, "middle")

    def walls(axis):
        anti = [b for b, a in zip(BLOCK_NAMES, odd_axis) if a == axis]
        bounce = [b for b, a in zip(BLOCK_NAMES, odd_axis) if a != axis]
        return f"anti-bounce-back for {', '.join(anti)}, bounce-back for {', '.join(bounce)}"

    return Drawing(
        DIAGRAM_WIDTH,
        bottom + 50 + 17 + 8,
        f"The unit square closed by walls on its four sides. The triangle x + y <= {corner_s} in"
        f" the corner (0, 0) holds the low state rho = {rho_in_s}, p = {p_in_s}; the rest of the"
        f" box holds rho = {rho_out_s}, p = {p_out_s}; the gas is at rest, u = v = 0. On the"
        f" left and right walls: {walls(0)}. On the top and bottom walls: {walls(1)}.",
        patterns(p),
        g,
    )
