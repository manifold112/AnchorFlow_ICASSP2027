"""Build the repository's SVG illustrations from geometry and paper-result CSVs."""

import csv
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
FONT = "Inter,Segoe UI,Arial,sans-serif"
NAVY, MINT, CORAL, MUTED = "#0c1830", "#69e6cb", "#ffad91", "#9babc4"


def text(x, y, content, size=16, fill="#eaf0fb", weight=400, **attributes):
    attrs = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attributes.items())
    return (f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" '
            f'font-weight="{weight}" font-family="{FONT}" {attrs}>{escape(content)}</text>')


def rect(x, y, w, h, fill, stroke="none", r=16):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}"/>'


def svg(name, width, height, body, description):
    OUT.mkdir(parents=True, exist_ok=True)
    source = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
              f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
              f'<title id="title">{escape(name)}</title><desc id="desc">{escape(description)}</desc>'
              + "".join(body) + '</svg>')
    (OUT / f"{name}.svg").write_text(source, encoding="utf-8")


def hero():
    b = [rect(0, 0, 1280, 574, NAVY, r=24)]
    for x in range(680, 1280, 40):
        b.append(f'<path d="M{x} 72V500" stroke="#1a2a45" stroke-width="1"/>')
    for y in range(100, 520, 40):
        b.append(f'<path d="M668 {y}H1230" stroke="#1a2a45" stroke-width="1"/>')
    b += [rect(56, 43, 156, 29, "#183444", r=14),
          text(73, 63, "AUDIO  /  RESEARCH", 11, MINT, 700, letter_spacing="1.1"),
          text(56, 152, "AnchorFlow", 78, "#f1f5fc", 700, letter_spacing="-3"),
          text(58, 204, "Beyond device invariance.", 31, MINT, 500),
          text(58, 261, "Scene-conditioned device flow modeling", 22, "#dae3f2"),
          text(58, 293, "for cross-device acoustic scene classification.", 22, "#dae3f2"),
          text(58, 348, "Model device variation. Preserve scene decisions.", 16, MUTED),
          rect(58, 389, 180, 38, "#1b3447", r=19),
          text(78, 414, "CLOSED-FORM FLOWS", 12, MINT, 700, letter_spacing="0.7"),
          rect(250, 389, 174, 38, "#252c41", r=19),
          text(268, 414, "TRAINING-TIME ONLY", 12, "#d8dfeb", 700, letter_spacing="0.5")]
    b += [text(697, 83, "ONE SCENE. MULTIPLE DEVICE CONDITIONS.", 11, MUTED, 600,
               letter_spacing="1.4")]
    # Contour ellipses are schematic distributions, not fitted experimental embeddings.
    for cx, cy, rx, ry, col in [(759, 300, 65, 95, "#63dac2"),
                               (1130, 220, 72, 60, "#ffad91")]:
        for factor, opacity in [(1.35, 0.12), (1.0, 0.25), (0.65, 0.48)]:
            b.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx*factor}" ry="{ry*factor}" '
                     f'fill="{col}" fill-opacity="0.035" stroke="{col}" '
                     f'stroke-opacity="{opacity}" stroke-width="1.5"/>')
    for j in range(7):
        offset = (j - 3) * 15
        x0, y0, x1, y1 = 759 + (j % 3 - 1) * 12, 300 + offset, 1130 + (j % 3 - 1) * 17, 220 + offset * .65
        b.append(f'<path d="M{x0} {y0}L{x1} {y1}" stroke="{MINT if j%2 else CORAL}" '
                 'stroke-opacity="0.55" stroke-width="1.5"/>')
        for t in [0, .25, .5, .75, 1]:
            px, py = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            b.append(f'<circle cx="{px}" cy="{py}" r="{4 if t in (0,1) else 2.4}" '
                     f'fill="{MINT if t<.5 else CORAL}" opacity="{.95 if t in (0,1) else .6}"/>')
    b += [text(712, 448, "SOURCE ANCHOR", 11, MINT, 700, letter_spacing="1"),
          text(1070, 333, "TARGET ANCHOR", 11, CORAL, 700, letter_spacing="1"),
          rect(864, 381, 239, 43, "#14243d", "#293d58", 10),
          text(884, 408, "z(t) = mu(t) + sigma(t) * r", 15, "#e8edf7", 500),
          text(879, 449, "Residual-preserving particle paths", 12, MUTED),
          '<path d="M56 479H1224" stroke="#283851"/>',
          text(58, 524, "NO LEARNED TRANSPORT", 12, "#a8b8ce", 600, letter_spacing="1.2"),
          text(432, 524, "NO ODE SOLVER", 12, "#a8b8ce", 600, letter_spacing="1.2"),
          text(760, 524, "UNCHANGED INFERENCE GRAPH", 12, "#a8b8ce", 600, letter_spacing="1.2")]
    svg("hero", 1280, 574, b, "AnchorFlow constructs linear, residual-preserving paths between same-scene device distributions. Distribution contours are schematic.")


def metrics():
    b = [rect(0, 0, 1280, 188, "#f4f6fa", r=20)]
    cards = [(16, "+3.31 pp", "UNSEEN DEVICE ACCURACY", "Average gain across 8 DCASE settings", "#077a69"),
             (438, "+3.22 pp", "HELD-OUT MICROPHONE RESPONSES", "CochlScene + MicIRP unseen split", "#c06142"),
             (860, "0", "ADDITIONAL INFERENCE PARAMETERS", "Original scene model retained at deployment", "#3b547b")]
    for x, value, title, desc, color in cards:
        b += [rect(x, 14, 404, 160, "#ffffff", "#e2e7ef", 14),
              text(x+23, 44, title, 10.3, color, 700, letter_spacing="0.9"),
              text(x+23, 105, value, 48, "#11233f", 700, letter_spacing="-1.5"),
              text(x+23, 141, desc, 12, "#62738d")]
    svg("highlights", 1280, 188, b, "Paper highlights: 3.31 percentage point average unseen-device gain; 3.22 point held-out microphone-response gain; no additional inference parameters. Gains are manuscript-reported.")


def architecture():
    b = [rect(0, 0, 1280, 660, "#f4f6fa", r=20),
          text(40, 49, "A TRAINING BRANCH. A SIMPLE DEPLOYMENT PATH.", 14, "#52657e", 700,
               letter_spacing="1.2"),
          text(40, 84, "Scene discrimination along structured device transitions", 27, "#10213b", 600)]
    b.append('<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="none" stroke="#698399" stroke-width="1.3"/></marker></defs>')

    def arrow(d, dashed=False):
        return f'<path d="{d}" fill="none" stroke="#698399" stroke-width="1.8" marker-end="url(#arrow)"' + (' stroke-dasharray="5 5"' if dashed else '') + '/>'

    def box(x, y, w, h, number, title, lines, color="#0d8072", fill="#ffffff"):
        b.extend([rect(x, y, w, h, fill, "#dce4ec", 14),
                  text(x+19, y+29, number, 11, color, 700, letter_spacing="1.0"),
                  text(x+19, y+60, title, 20, "#142b46", 600)])
        for i, line in enumerate(lines):
            b.append(text(x+19, y+89+i*22, line, 13, "#60758b"))

    box(40, 120, 205, 130, "INPUT", "Audio features", ["Single-source segments", "Scene y / device d"])
    box(292, 120, 245, 130, "ENCODER  f", "Representation h", ["Scene-discriminative features", "Normal base-loss gradients"])
    box(596, 120, 245, 130, "PREDICTOR  g", "Scene logits", ["Original scene classifier", "Shared trainable parameters"])
    box(908, 120, 330, 130, "OBJECTIVE", "Base scene loss", ["Cross-entropy on original features", "Updates encoder and predictor"], "#a25d43", "#fff9f5")
    b += [arrow("M245 185H292"), arrow("M537 185H596"), arrow("M841 185H908")]
    b += [arrow("M415 250V319", True), text(432, 292, "detach h", 12, "#95684b", 600)]
    box(40, 338, 250, 162, "01  /  ANCHORS", "Scene x device cells", ["EMA mean + raw second moment", "Same-scene device pairing", "Previously committed statistics"])
    box(338, 338, 250, 162, "02  /  GEOMETRY", "Closed-form paths", ["Fix standardized residual r", "Interpolate mean and scale", "Sample t and t + delta"])
    box(636, 338, 250, 162, "03  /  SHARED HEAD", "Deterministic g", ["Dropout / stochastic masks off", "BatchNorm statistics fixed", "Predictor gradients enabled"])
    box(934, 338, 304, 162, "04  /  REGULARIZATION", "Path-loss energy", ["Squared finite-difference slope", "First-order backpropagation", "L = L_base + lambda * E_AF"], "#a25d43", "#fff9f5")
    b += [arrow("M415 319H165V338", True), arrow("M415 319H463V338", True),
          arrow("M290 419H338"), arrow("M588 419H636"), arrow("M886 419H934"),
          arrow("M761 338V271H719V250", True),
          rect(40, 532, 1198, 83, "#e5f4ef", "#cce4dc", 13),
          text(61, 563, "COMMIT AFTER A SUCCESSFUL OPTIMIZER STEP", 11, "#0a7966", 700,
               letter_spacing="1.0"),
          text(61, 592, "Accumulate detached moments   →   optimizer step   →   commit anchors   →   next update", 15, "#315f55"),
          text(40, 640, "Deployment: audio → encoder → scene predictor. Device labels, anchors and auxiliary paths stay in training.", 13, "#536d83")]
    svg("architecture", 1280, 660, b, "Encoder and predictor follow the base training objective. A detached feature branch builds same-scene device paths and regularizes a shared deterministic predictor. Anchors commit after successful updates.")


def gains():
    with (ROOT / "benchmarks" / "paper_results.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    b = [rect(0, 0, 1280, 407, NAVY, r=20),
          text(40, 47, "GENERALIZATION GAINS CONCENTRATE ON UNSEEN DEVICES", 13, "#bdc9db", 700,
               letter_spacing="1.1"),
          text(40, 82, "Average accuracy improvement across four backbones", 25, "#f2f5fa", 600),
          text(40, 379, "Source: manuscript Table 1. Accuracy gains in percentage points; rounded table averages.", 12, MUTED)]
    # The manuscript prints gains from rounded average rows; preserve those
    # displayed values instead of recomputing a differently rounded raw mean.
    displayed_gains = {"dcase2020": {"seen": 1.37, "unseen": 3.24, "overall": 1.99},
                       "dcase2024": {"seen": 1.27, "unseen": 3.37, "overall": 1.97}}
    for x, protocol, label in [(40, "dcase2020", "DCASE 2020 / TAU20"),
                               (672, "dcase2024", "DCASE 2024 / TAU22")]:
        b.append(text(x, 128, label, 15, "#e4ebf7", 600))
        subset = [row for row in rows if row["protocol"] == protocol]
        for i, (metric, col, title) in enumerate([("seen", "#6685a9", "Seen"),
                                                ("unseen", MINT, "Unseen"),
                                                ("overall", CORAL, "Overall")]):
            if len(subset) != 4:
                raise ValueError("gain figure requires four backbones per protocol")
            gain = displayed_gains[protocol][metric]
            y = 167 + i * 56
            b += [text(x, y+17, title, 14, "#aebdd2"),
                  rect(x+88, y, 380, 26, "#1b2c45", r=6),
                  rect(x+88, y, gain/4*380, 26, col, r=6),
                  text(x+489, y+19, f"+{gain:.2f}", 18, col, 600)]
    svg("generalization", 1280, 407, b, "Manuscript results: average unseen-device gains of 3.24 and 3.37 percentage points on DCASE 2020 and 2024 respectively, larger than gains on seen devices.")


if __name__ == "__main__":
    hero()
    metrics()
    architecture()
    gains()
    print("Built 4 SVG assets")
