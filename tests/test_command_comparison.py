"""Matched one-step command provenance, constraint acceptance and offline math."""
import base64
import gzip
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest

import mujoco
import numpy as np

from cbf_demo.command_comparison import (build_command_comparison,
                                         command_comparison_script,
                                         comparison_math_fragment)
from cbf_demo.robot import RebotModel
from cbf_demo.robot_field import pair_clearance_constraints
from cbf_demo.scenes import START, GOAL, SURFACE_EPSILON, build_field, clutter_scene


class _EquationParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.equations = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'math-equation' in attrs.get('class', '').split() and 'data-tex' in attrs:
            self.equations.append(unescape(attrs['data-tex']))


class CommandComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = RebotModel(clutter_scene(), epsilon=SURFACE_EPSILON)
        cls.field = build_field(clutter_scene())
        cls.frames = [(0., START.copy()), (16.5, GOAL.copy())]
        cls.robot.set_q(START)
        cls.robot.data.qvel[:] = [.2, -.1, .3, .1, .2, -.1]
        cls.robot.data.qfrc_applied[:] = [1., 2., 3., 4., 5., 6.]
        cls.robot.data.time = 7.25
        mujoco.mj_forward(cls.robot.model, cls.robot.data)
        cls.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        cls.before = np.empty(mujoco.mj_stateSize(cls.robot.model, cls.state_spec))
        mujoco.mj_getState(cls.robot.model, cls.robot.data, cls.before, cls.state_spec)
        cls.h_before = cls.field.h.copy()
        cls.data = build_command_comparison(cls.robot, cls.field, cls.frames)
        cls.after = np.empty_like(cls.before)
        mujoco.mj_getState(cls.robot.model, cls.robot.data, cls.after, cls.state_spec)

    def test_thirteen_matched_nominals_at_identical_poses_and_margins(self):
        data = self.data
        self.assertEqual(len(data['command_names']), 13)
        self.assertEqual(data['margins_m'], [.025, .05, .1])
        self.assertEqual(len(data['frames']), len(self.frames))
        self.assertEqual(data['poisson_barrier_units'], 'm^2')
        self.assertEqual(data['distance_barrier_units'], 'm')
        self.assertFalse(data['matched_effective_clearance'])
        for frame, (time_s, q) in zip(data['frames'], self.frames):
            self.assertEqual(frame['time_s'], time_s)
            np.testing.assert_array_equal(frame['q_rad'], q)
            commands = frame['commands']
            self.assertEqual(len(commands), 13)
            np.testing.assert_allclose(commands[0]['nominal_rad_s'], np.clip(1.2*(GOAL-q), -.45, .45))
            for joint in range(6):
                for sign_index, sign in enumerate((1., -1.)):
                    expected = np.zeros(6); expected[joint] = sign*.45
                    item = commands[1+2*joint+sign_index]
                    np.testing.assert_array_equal(item['nominal_rad_s'], expected)
                    self.assertEqual(len(item['distance']), 3)
                    self.assertEqual(item['poisson']['all_rows'], len(self.robot.local_points))
                    self.assertTrue(all(result['all_rows'] == 50 for result in item['distance']))

    def test_every_accepted_distance_command_satisfies_all_original_pairs(self):
        for frame, (_, q) in zip(self.data['frames'], self.frames):
            geometry = pair_clearance_constraints(self.robot, q)
            self.assertTrue(geometry['valid'])
            for command in frame['commands']:
                for margin, result in zip(self.data['margins_m'], command['distance']):
                    if not result['success']:
                        continue
                    velocity = np.asarray(result['velocity_rad_s'])
                    self.assertLessEqual(float(np.max(np.abs(velocity))), .450002)
                    residual = geometry['gradients_q']@velocity + 2.*(geometry['distances_m']-margin)
                    self.assertGreaterEqual(float(residual.min()), -2e-6)
                    self.assertTrue(result['initial_margin_feasible'])
                    self.assertGreaterEqual(result['normalized_residual'], -2e-6)

    def test_no_simulator_integration_state_or_field_mutation(self):
        np.testing.assert_array_equal(self.after, self.before)
        np.testing.assert_array_equal(self.field.h, self.h_before)
        self.assertFalse(self.data['simulation_advanced'])
        self.assertFalse(self.data['physical_commands_sent'])

    def test_json_and_compressed_offline_payload_round_trip(self):
        json.dumps(self.data, allow_nan=False)
        script = command_comparison_script(self.data)
        payload = re.search(r"const payload='([^']+)';", script).group(1)
        decoded = json.loads(gzip.decompress(base64.b64decode(payload)))
        self.assertEqual(decoded, self.data)
        self.assertNotIn('__COMMAND_COMPARISON_GZIP__', script)
        self.assertIn('DecompressionStream', script)
        self.assertIn('rebot-comparison-command', script)
        self.assertNotIn('/home/', json.dumps(self.data))

    def test_all_four_comparison_equations_compile_in_official_katex(self):
        executable = shutil.which('node')
        if executable is None:
            self.skipTest('Node is needed for offline KaTeX syntax validation')
        parser = _EquationParser(); parser.feed(comparison_math_fragment())
        self.assertEqual(len(parser.equations), 4)
        asset = Path(__file__).resolve().parents[1]/'assets/katex/katex.min.js'
        program = """
          const fs=require('fs'),katex=require(process.argv[1]);
          const equations=JSON.parse(fs.readFileSync(0,'utf8'));
          const outputs=equations.map(tex=>katex.renderToString(tex,{throwOnError:true,
            output:'htmlAndMathml',trust:false,displayMode:true}));
          console.log(JSON.stringify({count:outputs.length,
            allMathml:outputs.every(s=>s.includes('<math')&&s.includes('katex-html'))}));
        """
        result = subprocess.run([executable, '-e', program, str(asset)],
                                input=json.dumps(parser.equations), text=True,
                                capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout), {'count': 4, 'allMathml': True})


if __name__ == '__main__':
    unittest.main()
