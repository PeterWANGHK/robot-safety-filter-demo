"""Source-backed, compact camera-to-field learning and benchmark roadmap.

This is an educational research plan. It runs no robot, camera, estimator,
predictor or safety controller. Method records are supplied as verified JSON.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from .math_rendering import offline_math_assets, math_render_script

_ASSETS = Path(__file__).resolve().parents[1] / "assets"

STAGES = [
    dict(name="1 · Observe", label="Wrist RGB-D → world points", goal="Locate each depth measurement using the robot pose at the image timestamp.",
         tex=r"{}^W\bar p={}^WT_E(q(t_k))\,{}^ET_C\begin{bmatrix}zK^{-1}[u,v,1]^\top\\1\end{bmatrix}",
         symbols="p̄: homogeneous point [x,y,z,1] · K: calibrated intrinsics · z: optical-axis depth in m · Tᴱ꜀: camera-to-tool calibration · q(tₖ): synchronized joints",
         practice="First replay a recorded RGB-D + CameraInfo + TF + joint-state bag. Overlay the transformed cloud on the official URDF at several poses.",
         check="Reject stale or missing transforms and invalid depth; rectify images before pinhole back-projection. Verify depth scale and optical-frame conventions.",
         source="https://github.com/marcoesposito1988/easy_handeye2", source_label="ROS 2 hand–eye calibration source"),
    dict(name="2 · Map", label="Points → observed geometry", goal="Keep observed free space, occupied surfaces and unseen space distinct.",
         tex=r"\mathcal M_t=\{\mathrm{free},\mathrm{occupied},\mathrm{unknown}\},\qquad D(p,t)\ [\mathrm m]",
         symbols="D: distance field · M: observed map · unknown: no trustworthy observation; a completed surface is a hypothesis",
         practice="Start with occupancy/OctoMap plus a CPU distance grid; compare a pinned nvblox TSDF/ESDF pipeline after measuring this Jetson's latency.",
         check="Mask the arm, gripper and carried object before fusion. A surface TSDF is not automatically a global Euclidean distance field. Occlusion and depth dropout must not clear unknown space.",
         source="https://moveit.picknik.ai/main/doc/examples/perception_pipeline/perception_pipeline_tutorial.html", source_label="MoveIt 3D perception pipeline"),
    dict(name="3 · Predict", label="Observed motion → occupied tubes", goal="Predict moving obstacles with an uncertainty envelope rather than one certain future position.",
         tex=r"\mathcal O_j(t+\tau)=\widehat{\mathcal O}_j(t+\tau)\oplus\mathbb B_{\epsilon_j(\tau)}",
         symbols="τ: prediction horizon in s · εⱼ: validated error bound in m · ⊕: Minkowski sum · B: uncertainty ball",
         practice="Begin with constant-velocity tracking and bounded acceleration. Compare learned forecasting only after testing its out-of-distribution uncertainty coverage.",
         check="Prediction accuracy alone does not establish safety. Evaluate missed detections, changes of direction, blind-side entry, camera motion and stale tracks.",
         source="https://github.com/StanfordASL/Trajectron-plus-plus", source_label="Trajectory prediction reference; pedestrian/vehicle domain, not an arm safety certificate"),
    dict(name="4 · Lift", label="3D map → whole-arm state constraints", goal="Evaluate predicted separation for every relevant robot body, the camera bracket and the held object.",
         tex=r"b_i(x,t;\tau)=D_{\rm lower}(p_i(q_\tau),t+\tau)-r_i-\delta_i,\quad x=(q,\dot q)",
         symbols="qτ: candidate predicted configuration · D_lower: conservative distance estimate · pᵢ: body sample/sphere centre · rᵢ: coverage radius · δᵢ: model margin",
         practice="Retain reachability → collision-aware planner → safety filter → joint controller. Add separate self-collision, joint-limit and payload constraints.",
         check="The rendered 3D field is a slice. A six-joint arm already has a 12-dimensional position–velocity state; map belief, obstacle states and time add dimensions.",
         source="https://lavalle.pl/planning/", source_label="Configuration space and planning under uncertainty"),
    dict(name="5 · Filter", label="Candidate motion → feasible command", goal="Use an explicit dynamics model and input bounds to decide which motions are admissible.",
         tex=r"L_f b_i+L_g b_i\,u+\partial_t b_i\ge-\alpha_i(b_i)",
         symbols="f,g: dynamics · u: chosen velocity, acceleration or torque input · ∂ₜb: changing environment · α: barrier gain",
         practice="Reproduce the current velocity CBF first. Then compare predictive MPC-CBF or backup filtering with actuator saturation and a tested stopping policy.",
         check="A position barrier usually needs higher relative-degree treatment for acceleration/torque input. Differentiability, estimation-error bounds, feasibility and tracking assumptions must be checked.",
         source="https://arxiv.org/abs/1609.06408", source_label="Ames et al., CBF-QP foundations, IEEE TAC 2017"),
    dict(name="6 · Validate", label="Closed-loop trials → evidence", goal="Measure collisions against independent ground truth, including hazards the wrist camera cannot observe.",
         tex=r"d_{\min}=\min_{t,i,j}d(\mathcal R_i(q(t)),\mathcal O_j(t))",
         symbols="Rᵢ: physical robot geometry · Oⱼ: ground-truth obstacle · t: execution time · d_min: minimum execution clearance",
         practice="Run matched seeds, tasks, observations, controller budgets and hardware for each baseline. Keep training, tuning and held-out scenes separate.",
         check="Report collision and task success together; also report latency, uncertainty coverage, QP infeasibility, braking failures, torque limits and minimum clearance. Safety and closed-loop stability are different claims.",
         source="https://github.com/fishbotics/robometrics", source_label="Robometrics evaluation source"),
]

FOUNDATIONS = [
    ("Rigid transforms, FK and Jacobians", "Modern Robotics, chapters 2–6", "Transform a wrist depth point into the base frame; check J with finite differences.", "https://modernrobotics.northwestern.edu/nu-gm-book-resource/"),
    ("Dynamics, statics and control", "Modern Robotics, chapters 8, 11–12", "Compute payload gravity torque and compare commanded torque with saturation and tracking error.", "https://modernrobotics.northwestern.edu/nu-gm-book-resource/"),
    ("Configuration space and planning", "LaValle, chapters 4–5, 11–12, 14", "Explain why a reachable TCP does not imply an elbow-safe path; model unknown space explicitly.", "https://lavalle.pl/planning/"),
    ("RGB-D geometry and hand–eye calibration", "ROS 2 TF, CameraInfo, depth scale and AX=XB", "Reproject a held-out target at multiple poses and measure spatial error and time alignment.", "https://github.com/marcoesposito1988/easy_handeye2"),
    ("CBF, CLF and constrained optimization", "Ames et al., IEEE TAC 2017; then higher-order and robust CBFs", "Derive the constraint for the selected input, check units, and deliberately test infeasible cases.", "https://arxiv.org/abs/1609.06408"),
    ("Distance mapping and uncertainty", "Occupancy → TSDF → ESDF; error bounds and prediction", "Measure distance error, false-safe regions and map age before using the field in a filter.", "https://nvidia-isaac-ros.github.io/v/release-4.6/repositories_and_packages/isaac_ros_nvblox/index.html"),
]

BENCHMARKS = [
    dict(title="MotionBenchMaker", venue="IEEE RA-L 2022", paper="https://arxiv.org/abs/2112.06402", code="https://github.com/KavrakiLab/motion_bench_maker", license="BSD-3-Clause",
         use="Static clutter and narrow-passage tasks; generate matched start/goal/environment sets.",
         transfer="Original ROS 1 tooling and robot-specific datasets need a ROS 2/reBot adapter. ReBot-retargeted tasks are a new benchmark, not directly comparable paper scores."),
    dict(title="MπNets + Robometrics", venue="CoRL 2022", paper="https://arxiv.org/abs/2210.12209", code="https://github.com/NVlabs/motion-policy-networks", license="MIT; check dataset terms separately",
         use="Single-depth/partial-point-cloud planning, data generation and a shared motion evaluator.",
         transfer="Franka-trained policy and collision geometry do not directly transfer to B601. Original data generation uses Isaac Sim; use the released tasks/evaluator or your own MuJoCo generator on this Jetson."),
    dict(title="Neural MP", venue="IEEE/RSJ IROS 2025", paper="https://arxiv.org/abs/2409.05864", code="https://github.com/mihdalal/neuralmotionplanner", license="No repository-wide licence verified; inspect before code reuse",
         use="A camera-to-motion policy plus test-time optimization as a learned nominal-planner reference.",
         transfer="Published deployment is Franka with multiple external RealSense cameras; pretrained weights are not a wrist-camera B601 controller. Training and compiled dependencies require adaptation."),
]

EXPERIMENTS = [
    ("Geometry oracle", "Exact simulated scene; static boxes and mesh clutter", "Planner only; current Poisson CBF; distance CBF; ESDF-CBF", "Collision, success, minimum clearance, path length, infeasible steps"),
    ("Wrist RGB-D", "Matched scenes rendered through a calibrated wrist camera", "Same planners and filters; observed map instead of oracle map", "Depth/map error, unseen-space false-safe rate, self-mask errors, end-to-end latency"),
    ("Moving hazards", "Crossing obstacles, sudden direction changes and blind-side entry", "Static-map filter; constant-velocity prediction; robust predictive filter", "Contact, time-to-collision error, prediction coverage, interventions and braking distance"),
    ("Manipulation load", "Reach, grasp, carry and insert with held-out payloads", "Add camera/bracket/payload geometry and dynamics limits", "Collision, grasp retention, joint torque/current, tracking error, task completion"),
    ("Stress and transfer", "Depth dropout, TF delay, calibration error; then mobile base", "Ablate prediction, uncertainty, memory and backup policy", "p50/p95/p99 latency, missed deadlines, energy, stopping success, base stability"),
]


def build_research_roadmap(methods, destination, *, demo_href="index.html"):
    """Write an offline educational page with primary-source paper/code links."""
    records = methods.get("papers", methods.get("methods", [])) if isinstance(methods, dict) else methods
    if not isinstance(records, list):
        raise ValueError("Expected a list of literature method records")
    payload = dict(stages=STAGES, methods=records)
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    h = html.escape
    foundation_html = "".join(f'<details><summary>{i+1}. {h(name)}</summary><p><a href="{h(url, quote=True)}">{h(reading)}</a></p><p class="exercise">Practice: {h(exercise)}</p></details>' for i,(name,reading,exercise,url) in enumerate(FOUNDATIONS))
    benchmark_html = "".join(f'<article><p class="tag">{h(b["venue"])}</p><h3>{h(b["title"])}</h3><p>{h(b["use"])}</p><p class="limit">{h(b["transfer"])}</p><p><a href="{b["paper"]}">Paper</a> · <a href="{b["code"]}">Code</a></p><small>{h(b["license"])}</small></article>' for b in BENCHMARKS)
    experiment_html = "".join('<tr>'+''.join(f'<td>{h(v)}</td>' for v in row)+'</tr>' for row in EXPERIMENTS)
    supporting = methods.get('supporting_reading', []) if isinstance(methods, dict) else []
    supporting_html = ''.join('<details><summary>'+h(p['title'])+'</summary><p>'+h(str(p.get('venue','')))+' · '+h(str(p.get('year','')))+'</p><p>'+h(p.get('role',''))+'</p><p>'+h(p.get('code_status',''))+'</p><p><a href="'+h(p['paper_url'],quote=True)+'">Paper</a>'+(' · <a href="'+h(p['code_url'],quote=True)+'">Code reference</a>' if p.get('code_url') else '')+'</p></details>' for p in supporting)
    css = (_ASSETS/"research_roadmap.css").read_text()
    js = (_ASSETS/"research_roadmap.js").read_text()
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Wrist RGB-D → predictive arm safety · Research roadmap</title><style>'''+css+'''</style></head><body><main>
<a class="back" href="'''+h(demo_href, quote=True)+'''">← 3D arm and field demo</a>
<header><p class="eyebrow">Research roadmap · checked 28 November 2026</p><h1>From the wrist camera to a predictive safety field</h1><p>Learn the pieces, reproduce the baselines, then test a new method on the fixed-base B601.</p></header>
<section aria-labelledby="pipeline-title"><h2 id="pipeline-title">Follow a measurement to a control constraint</h2><div id="pipeline-steps" role="group" aria-label="Camera-to-field stages"></div>
<div class="pipeline-layout"><article id="stage-card" aria-live="polite"><h3 id="stage-label"></h3><p id="stage-goal"></p><div id="stage-equation" class="math-equation"></div><p id="stage-symbols" class="symbols"></p><p id="stage-practice" class="exercise"></p><p id="stage-check" class="limit"></p><a id="stage-source"></a></article>
<aside><h3>What “high-dimensional” means</h3><label>Robot <select id="state-robot"><option value="6">Fixed-base arm · 6 joints</option><option value="9">Planar mobile base + arm · 9 configuration variables</option></select></label><p id="state-count" class="state-count"></p><p>Obstacle trajectories, map uncertainty, payload state and time add variables. A 3D view shows a spatial slice. A wheeled base also restricts admissible velocities; its coordinates are not independent actuators.</p><p class="proposal">Candidate research question: can an uncertainty-aware, time-dependent distance field reduce collisions under wrist-camera occlusion at the same task success and latency budget?</p><small>A proposed study, not a validated new algorithm or novelty claim.</small></aside></div></section>
<section id="methods"><h2>Methods with inspectable implementations</h2><p>Use mapping to estimate geometry, planning to seek a route, and filtering to constrain the command. Code availability is recorded separately from a safety claim.</p>
<div class="searchbar"><label>Find a method <input id="paper-search" type="search" placeholder="ESDF, CBF, prediction, robot distance…"></label><label>Role <select id="paper-role"><option value="all">All roles</option><option value="mapping">Mapping / representation</option><option value="filter">Safety filter / control</option><option value="planning">Planning / policy</option><option value="reachability">Reachability</option></select></label><output id="paper-count"></output></div><div id="method-cards" class="cards"></div></section>
<section><h2>Learn in this order</h2><div class="foundations">'''+foundation_html+'''</div><h3>Predictive foundations and further reading</h3>'''+supporting_html+'''</section>
<section><h2>Benchmark starting points</h2><div class="cards">'''+benchmark_html+'''</div><p>Use the published protocols as references. Retargeting tasks to a six-joint B601 changes robot reachability, dynamics and observation coverage; report those changes explicitly.</p></section>
<section><h2>A staged benchmark for this arm</h2><div class="table-wrap"><table><thead><tr><th>Stage</th><th>Environment / observations</th><th>Matched comparisons</th><th>Measurements</th></tr></thead><tbody>'''+experiment_html+'''</tbody></table></div>
<p class="proposal">Start with a fixed nominal planner and change only the filter. Then hold the filter fixed and compare planners. Use independent execution collision checks, held-out scenes and paired random seeds; report confidence intervals across trials. The current pose replay is a teaching artifact, not this closed-loop benchmark.</p></section>
<section><h2>First wrist-camera trial</h2><ol><li>Choose the camera driver and record RGB, rectified depth, CameraInfo, joint states, TF and timestamps.</li><li>Calibrate camera-to-tool pose; replay a bag and check cloud alignment on held-out views.</li><li>Mask robot, mount and payload; preserve an explicit unknown-space mask and map age.</li><li>Build the observed occupancy/ESDF map in simulation, compare it with hidden ground truth, then enable a velocity filter in simulated closed loop.</li><li>Add tracked moving objects, uncertainty and stopping constraints; only then compare a learned nominal policy.</li></ol>
<p>The camera brand, intrinsics, tool-to-camera transform and mount geometry are still required for real observations. This page runs no camera or robot commands.</p>
<details><summary>Jetson software fit and source notes</summary><p>The inspected host is an AGX Orin on Ubuntu 24.04 with ROS 2 Jazzy and L4T R39.2. NVIDIA’s <a href="https://nvidia-isaac-ros.github.io/v/release-4.6/repositories_and_packages/isaac_ros_nvblox/index.html">Isaac ROS 4.6 nvblox documentation</a> lists Jazzy and Orin/JetPack 7.2; release 5.0 targets ROS 2 Lyrical. Pin the supported dependency set before installing. No new mapping stack was installed or benchmarked for this roadmap.</p><p>cuRobo’s current API differs from its v1 releases. Use a pinned release and reproduce its robot/sphere validation before comparing. Vendor speed measurements are not measurements on this arm or this Jetson.</p><p>Literature cards link to primary papers, author projects and source repositories. A source review is not an independent reproduction. Licences and dependency constraints still need review before integrating code.</p></details></section>
</main><script id="roadmap-data" type="application/json">'''+encoded+'''</script>'''+offline_math_assets()+"<script>"+js+"</script>"+math_render_script()+"</body></html>"
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page)
    return dict(method_count=len(records), pipeline_stages=len(STAGES), benchmark_references=len(BENCHMARKS), educational_only=True)
