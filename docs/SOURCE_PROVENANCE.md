# Source provenance and licence boundaries

This document records upstream inputs and local adaptations for the static
B601 safety-filter demonstration. Component licences remain distinct; a licence
for newly written demonstration code does not replace the upstream terms below.

## B601 robot description and derived geometry

- Upstream: [Yang-Ci/ReBot_Arm_DigitalTwin_DM](https://github.com/Yang-Ci/ReBot_Arm_DigitalTwin_DM).
- Pinned revision: `914c763c4adffb5f5744c43ecffbf6199c282bd6`.
- Original MJCF:
  [`reBotArm_ros2_DM/src/rebotarm_mujoco/models/rebotarm_b601_colored.xml`](https://github.com/Yang-Ci/ReBot_Arm_DigitalTwin_DM/blob/914c763c4adffb5f5744c43ecffbf6199c282bd6/reBotArm_ros2_DM/src/rebotarm_mujoco/models/rebotarm_b601_colored.xml).
- Mesh source:
  [`reBotArm_ros2_DM/src/rebotarm_bringup/description/meshes`](https://github.com/Yang-Ci/ReBot_Arm_DigitalTwin_DM/tree/914c763c4adffb5f5744c43ecffbf6199c282bd6/reBotArm_ros2_DM/src/rebotarm_bringup/description/meshes).
- Licence: [upstream CERN-OHL-W-2.0](https://github.com/Yang-Ci/ReBot_Arm_DigitalTwin_DM/blob/914c763c4adffb5f5744c43ecffbf6199c282bd6/LICENSE),
  copied without alteration to [CERN-OHL-W-2.0.txt](../LICENSES/CERN-OHL-W-2.0.txt).

**Modification notice, 26 November 2026:** the preliminary experiment adapts the
description for a fixed-base six-joint simulation with fixed open fingers,
conservative body-local enclosing boxes, surface samples, explicit synthetic
obstacles, and independently defined saturated PD control. The model's joint
axes and inertials inform the static-load calculation. Browser geometry and
distance envelopes are derived from these proxies; they are not an exact
physical scan or a certified collision envelope.

Redistributed descriptions and model-derived proxy geometry retain the
upstream licence and notices. The complete adaptation source and a source
location must accompany distributions as required by those terms. The pinned
upstream links identify the preferred original model sources; the demo's
published source identifies its modifications. Independently written algorithms
and separately licensed components are identified separately rather than
relicensing all upstream material under one permissive label.

The unrelated ROS controller and low-level hardware SDK repositories are not
needed to view this page and are not part of its static runtime. Their presence
in a development workstation does not make them part of this publication.

## Rendering and plotting components

| Component | Version used | Provenance and retained notice |
|---|---|---|
| Plotly.js | 3.3.0, bundled by Plotly.py 6.5.0 | [Official tagged source](https://github.com/plotly/plotly.js/tree/v3.3.0), [MIT notice](../LICENSES/PLOTLY-JS-MIT.txt). Preserve the bundle's copyright banner, including its 2012–2025 notice. |
| KaTeX | 0.16.22 | [Official npm archive](https://registry.npmjs.org/katex/-/katex-0.16.22.tgz), [MIT notice](../LICENSES/KATEX-MIT.txt). Vendored metadata records archive and file hashes; inline HTML retains the notice. |
| SciencePlots | 2.2.0 | [Official source](https://github.com/garrettj403/SciencePlots), [MIT notice](../LICENSES/SCIENCEPLOTS-MIT.txt). Used by the Python figure-generation workflow; not a browser control component. |

KaTeX's archive SHA-256 is
`e9e0d167db3175481cbadaff38e8d90b130f6a3ddb451a47e43c577fd511f365`.
Its archive was checked against the official npm SHA-1 and SHA-512 integrity
values. Only minified JavaScript/CSS, WOFF2 fonts, the licence and provenance
metadata are retained. Fonts can be embedded as data URLs for portable HTML.

MuJoCo is a simulator dependency of the Python experiment, not a browser
simulator embedded in this page. See its
[official source and licence](https://github.com/google-deepmind/mujoco).
Generated plots and recorded synthetic results should retain their scenario,
version and assumption metadata; a successful recorded episode is not a claim
about unrestricted hardware operation.

## Method attribution

The environment method is inspired by Wilkinson et al.,
[arXiv:2604.21189v1](https://arxiv.org/html/2604.21189v1). The robot-centred
nearest-pair comparison and hierarchy adaptations are described in
[MATH_COMPARISON.md](MATH_COMPARISON.md). No paper PDF or unreleased author
implementation is redistributed here. Neither the article nor an upstream
logo is used as evidence of endorsement.
