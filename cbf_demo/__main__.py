import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import platform
import sys

import numpy as np

from .experiment import prepare, plan_path, run_episode, save_run, plot_runs, plot_preparation
from .scenes import START, clutter_scene


def main():
    parser = argparse.ArgumentParser(description='Simulation-only B601 Poisson-CBF experiment')
    sub = parser.add_subparsers(dest='command', required=True)
    view = sub.add_parser('visualize', help='3D field views and frozen-state filter queries; no physics stepping')
    view.add_argument('--run', type=Path, help='Saved run containing configuration.json (e.g. results/fixed-base-final/seed-7)')
    view.add_argument('--output', type=Path)
    load = sub.add_parser('load', help='Frozen-pose payload holding estimate; no physics stepping')
    load.add_argument('--payload', type=float, default=.5, help='Point-mass payload, kg')
    load.add_argument('--com', type=float, nargs=3, default=[0.,0.,0.], metavar=('X','Y','Z'),
                      help='Payload COM offset in TCP coordinates, metres')
    load.add_argument('--torque-fraction', type=float, default=.7, help='Usable fraction of model torque caps')
    load.add_argument('--q', type=float, nargs=6, default=START.tolist(), help='Six joint angles, radians')
    load.add_argument('--output', type=Path)
    for name in ('demo', 'benchmark', 'reachability', 'plan'):
        p = sub.add_parser(name)
        p.add_argument('--seed', type=int, default=7)
        p.add_argument('--map-samples', type=int, default=250)
        p.add_argument('--spacing', type=float, default=.025, help='Poisson grid spacing, metres')
        p.add_argument('--output', type=Path)
        if name in ('demo','benchmark'):
            p.add_argument('--plant', choices=['dynamics','kinematic'], default='dynamics')
            p.add_argument('--seconds', type=float, default=35.)
        if name == 'demo':
            p.add_argument('--mode', choices=['hierarchy','filter-only','unfiltered'],default='hierarchy')
            p.add_argument('--viewer', action='store_true')
            p.add_argument('--ros', action='store_true')
            p.add_argument('--hold', action='store_true')
        if name == 'benchmark':
            p.add_argument('--seeds', type=int, nargs='+', default=[7,11,19])
    args = parser.parse_args()
    if args.command == 'load':
        from .robot import RebotModel
        from .load_model import analyze_static_load
        robot = RebotModel(obstacles=clutter_scene(), epsilon=.025)
        try:
            report = analyze_static_load(robot, args.q, payload_kg=args.payload,
                                          com_offset=args.com, headroom_fraction=args.torque_fraction)
        except ValueError as error:
            parser.error(str(error))
        destination = args.output or Path('results')/('load-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        destination.mkdir(parents=True, exist_ok=True)
        (destination/'static_load.json').write_text(json.dumps(report, indent=2,allow_nan=False)+'\n')
        print('Frozen-pose model only: own-arm gravity + payload holding torque. No physics advanced.')
        print(f'Payload: {args.payload:g} kg; weight: {np.linalg.norm(report["weight_force_world_n"]):.3f} N; usable model torque: {args.torque_fraction:.0%}.')
        print('joint  arm N.m  payload N.m  total N.m  budget N.m  used')
        for i in range(6):
            print(f'J{i+1}  {report["tau_robot_nm"][i]:9.3f} {report["tau_payload_nm"][i]:11.3f} {report["tau_total_nm"][i]:10.3f} {report["caps_nm"][i]:11.3f} {report["utilization"][i]:8.1%}')
        interval = report['payload_interval']
        print('Static interval: '+json.dumps(interval))
        print(f'Report: {(destination/"static_load.json").resolve()}')
        return 0
    if args.command == 'visualize':
        from .field_plots import visualize_saved, plot_field_3d
        destination = args.output or Path('results')/('visualization-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        print(f'Visualization output: {destination.resolve()}', flush=True)
        if args.run is not None:
            if not (args.run/'configuration.json').is_file():
                parser.error('--run must contain configuration.json; select a seed folder for a benchmark')
            visualize_saved(args.run, destination)
        else:
            robot, field, _, reach, _, target = prepare()
            plot_preparation(field, reach, target, destination)
            plot_field_3d(field, robot, destination, target=target)
        print('Saved field_3d.html (parallel fields + command comparison), SciencePlots PNG/PDF, field/command data and visualization.json. No physics advanced.', flush=True)
        return 0
    if args.map_samples < 0 or not np.isfinite(args.spacing) or args.spacing <= 0:
        parser.error('map samples must be nonnegative; spacing must be positive')
    destination = args.output or Path('results')/(args.command+'-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    destination.mkdir(parents=True, exist_ok=True)
    print(f'Output: {destination.resolve()}',flush=True)
    seeds = args.seeds if args.command == 'benchmark' else [args.seed]
    all_summaries, plotted = [], []
    for seed in seeds:
        seed_results = []
        print(f'Preparing actual B601 model, static 3D field and reachability seeds (seed {seed})...',flush=True)
        robot, field, checker, reach, goal, target = prepare(seed,args.map_samples,args.spacing)
        seed_dir = destination/f'seed-{seed}' if args.command=='benchmark' else destination
        seed_dir.mkdir(parents=True,exist_ok=True)
        reach.save(seed_dir/'reachability.npz')
        metadata=dict(paper='https://arxiv.org/abs/2604.21189v1',platform=platform.platform(),
                      python=sys.version.split()[0],seed=seed,model=robot.geometry_metadata,
                      field=field.solve_info,grid_shape=field.shape,spacing_m=field.spacing,
                      grid_origin_m=field.origin.tolist(), obstacles=clutter_scene(),
                      requested_inflation_m=field.inflation,voxel_padding_m=field.padding,
                      reachability_accepted=reach.sample_count,
                      sampled_position_voxels=len(reach.voxel_keys),
                      reachability_attempted=reach.attempted_samples,
                      task_seed_included=True,target_xyz=target.tolist(),
                      start_q=START.tolist(),goal_q=goal.tolist(),
                      cbf_alpha=2.,command_speed_bound_rad_s=.45,h_reserve_m2=2e-5,
                      osqp_rho=1.,osqp_absolute_tolerance=1e-7,
                      orientation_certified=False,hardware_actuated=False)
        (seed_dir/'configuration.json').write_text(json.dumps(metadata,indent=2)+'\n')
        plot_preparation(field,reach,target,seed_dir)
        print(f'Field: {field.solve_info["seconds"]:.3f}s; {len(robot.local_points)} body samples; reach map {reach.sample_count}/{reach.attempted_samples} accepted.',flush=True)
        if args.command == 'reachability':
            from .field_plots import plot_field_3d
            plot_field_3d(field, robot, seed_dir, target=target)
            print('Saved position-only sampled reach map (known goal seed included).',flush=True)
            return 0
        result = plan_path(robot,checker,goal,seed)
        record=asdict(result)
        record['path']=result.path.tolist() if result.path is not None else None
        (seed_dir/'plan.json').write_text(json.dumps(record,indent=2)+'\n')
        print(f'Planner: {result.reason}, {result.elapsed_s:.3f}s, {result.collision_checks} checks.',flush=True)
        if args.command == 'plan':
            from .field_plots import plot_field_3d
            plot_field_3d(field, robot, seed_dir, path=result.path, target=target)
            return 0 if result.success else 2
        if not result.success:
            print('No safe planned route; no motion executed.',file=sys.stderr)
            return 2
        modes=['unfiltered','filter-only','hierarchy'] if args.command=='benchmark' else [args.mode]
        for mode in modes:
            print(f'Running {mode}, {args.plant}...',flush=True)
            rows, summary=run_episode(robot,field,goal,target,mode=mode,path=result.path,
                                       plant=args.plant,seconds=args.seconds,
                                       viewer=getattr(args,'viewer',False),ros=getattr(args,'ros',False),
                                       hold=getattr(args,'hold',False))
            summary['seed']=seed
            save_run(seed_dir/mode,rows,summary,result.path)
            all_summaries.append(summary)
            seed_results.append((rows, summary))
            if seed==seeds[0]:plotted.append((rows,summary))
            print(f'{mode}: {summary["status"]}; min clearance={summary["minimum_clearance_m"]:.4f}m; error={summary["final_goal_error_m"]:.4f}m; filter interventions={summary["filter_interventions"]}.',flush=True)
        from .field_plots import plot_field_3d
        plot_field_3d(field, robot, seed_dir,
                      results=seed_results,
                      path=result.path, target=target)
    (destination/'summary.json').write_text(json.dumps(all_summaries,indent=2)+'\n')
    plot_runs(plotted,destination/'comparison.png')
    print('Saved CSV, JSON, planned path, reach map and comparison.png.',flush=True)
    return 0 if args.command=='benchmark' or all(s['success'] for s in all_summaries) else 2


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Simulation interrupted; no physical driver was used.',file=sys.stderr)
        raise SystemExit(130)
