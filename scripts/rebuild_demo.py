"""Regenerate the parallel interactive comparison from the published poses.

Run from the repository root after fetching the original licensed model.
This runs local QP queries only; it never advances physics or starts ROS.
"""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('REBOT_ROOT', str(ROOT/'vendor'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from plotly.offline import get_plotlyjs
from cbf_demo.robot import RebotModel
from cbf_demo.scenes import build_field, clutter_scene
from cbf_demo.comparison_view import (build_comparison_data, comparison_fragment,
                                     comparison_script, plot_field_comparison)
from cbf_demo.command_comparison import (build_command_comparison, command_comparison_fragment,
                                        command_comparison_script, comparison_math_fragment,
                                        plot_command_comparison)
from cbf_demo.math_rendering import offline_math_assets, math_render_script
from cbf_demo.derivation_data import build_derivation_data
from cbf_demo.derivation_tour import derivation_tour_script


def main():
    out = ROOT/'build'
    out.mkdir(exist_ok=True)
    poses = json.loads((ROOT/'data'/'recorded_poses.json').read_text())
    frames = [(row['time_s'], row['q_rad']) for row in poses['frames']]
    robot = RebotModel(obstacles=clutter_scene(), epsilon=.025)
    field = build_field(clutter_scene(), spacing=.025, inflation=.035)
    comparison = build_comparison_data(robot, field, frames)
    commands = build_command_comparison(robot, field, frames)
    derivation = build_derivation_data(robot, field, frames, command_data=commands)
    page = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B601 robot safety filter</title><style>body{max-width:1500px;margin:24px auto;padding:0 20px;font:17px/1.5 Georgia,serif;color:#273340}.note{background:#f4f6f9;padding:16px}td,th{padding:6px}.math-equation{overflow-x:auto}</style></head><body><h1>Robot safety filter demo · B601</h1>'
    page += comparison_fragment(comparison)+command_comparison_fragment()
    page += '<script>'+get_plotlyjs()+'</script>'+offline_math_assets()
    page += comparison_script(comparison)+command_comparison_script(commands)+derivation_tour_script(derivation)+math_render_script()+'</body></html>'
    (out/'index.html').write_text(page)
    (out/'field_comparison_data.json').write_text(json.dumps(comparison, separators=(',', ':'), allow_nan=False))
    (out/'command_comparison_data.json').write_text(json.dumps(commands, separators=(',', ':'), allow_nan=False))
    (out/'derivation_data.json').write_text(json.dumps(derivation, separators=(',', ':'), allow_nan=False))
    plot_field_comparison(comparison, out/'field_comparison.png')
    plot_command_comparison(commands, out/'command_comparison.png')
    print('Wrote build/index.html and SciencePlots PNG/PDF figures. No physics advanced.')


if __name__ == '__main__':
    main()
