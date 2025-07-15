import json
import numpy as np
from scipy.stats import truncnorm
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import time
from typing import Dict, Any, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# For power system optimization - you'll need to install pyomo and appropriate solvers
try:
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory

    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False
    print("Warning: Pyomo not available. Install with: pip install pyomo")

# Set random seed for reproducibility
np.random.seed(123)


def parse_commandline():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate OPF training data')
    parser.add_argument('--netname', '-n', type=str, default='nesta_case14_ieee',
                        help='The input network name')
    parser.add_argument('--output', '-o', type=str, default='traindata_ext',
                        help='The output name')
    parser.add_argument('--lb', type=float, default=0.8,
                        help='The lb (in %) of the load interval')
    parser.add_argument('--ub', type=float, default=1.2,
                        help='The ub (in %) of the load interval')
    parser.add_argument('--step', type=float, default=0.0001,
                        help='The step size resulting in a new load x + step')
    parser.add_argument('--nperm', type=int, default=10,
                        help='The number of load permutations for each load scale')
    return parser.parse_args()


def parse_matpower_file(filename: str) -> Dict[str, Any]:
    """
    Parse MATPOWER .m file format (simplified version)
    This is a simplified parser - for production use, consider using MATPOWER directly
    or a more robust parser like pandapower
    """
    # This is a placeholder - in practice, you'd use a proper MATPOWER parser
    # or convert files to JSON format first
    data = {
        'bus': {},
        'gen': {},
        'branch': {},
        'load': {}
    }

    # For demonstration, create a simple 14-bus system structure
    # In practice, you'd parse the actual MATPOWER file
    if '14' in filename or 'nesta_case14' in filename:
        # Simplified 14-bus system structure
        for i in range(1, 15):
            data['bus'][str(i)] = {
                'vmin': 0.94,
                'vmax': 1.06,
                'bus_type': 1 if i > 1 else 3
            }

        # Add some generators
        data['gen']['1'] = {'gen_bus': 1, 'pmin': 0.0, 'pmax': 232.4, 'qmin': -10.0, 'qmax': 10.0}
        data['gen']['2'] = {'gen_bus': 2, 'pmin': 0.0, 'pmax': 40.0, 'qmin': -40.0, 'qmax': 50.0}
        data['gen']['3'] = {'gen_bus': 3, 'pmin': 0.0, 'pmax': 0.0, 'qmin': 0.0, 'qmax': 40.0}
        data['gen']['6'] = {'gen_bus': 6, 'pmin': 0.0, 'pmax': 0.0, 'qmin': -6.0, 'qmax': 24.0}
        data['gen']['8'] = {'gen_bus': 8, 'pmin': 0.0, 'pmax': 0.0, 'qmin': -6.0, 'qmax': 24.0}

        # Add some loads
        load_data = [(2, 21.7, 12.7), (3, 94.2, 19.0), (4, 47.8, -3.9), (5, 7.6, 1.6)]
        for i, (bus, pd, qd) in enumerate(load_data):
            data['load'][str(i + 1)] = {'bus': bus, 'pd': pd, 'qd': qd}

        # Add some branches
        branch_data = [
            (1, 2, 0.01938, 0.05917, 0.0528, 9.9),
            (1, 5, 0.05403, 0.22304, 0.0492, 6.7),
            (2, 3, 0.04699, 0.19797, 0.0438, 4.8),
            (2, 4, 0.05811, 0.17632, 0.0340, 6.7),
            (2, 5, 0.05695, 0.17388, 0.0346, 6.7),
        ]
        for i, (f_bus, t_bus, r, x, b, rate) in enumerate(branch_data):
            data['branch'][str(i + 1)] = {
                'f_bus': f_bus, 't_bus': t_bus, 'br_r': r, 'br_x': x,
                'b_fr': b / 2, 'b_to': b / 2, 'g_fr': 0.0, 'g_to': 0.0,
                'rate_a': rate
            }
    else:
        # Create a minimal system for unknown cases
        data['bus']['1'] = {'vmin': 0.94, 'vmax': 1.06, 'bus_type': 3}
        data['gen']['1'] = {'gen_bus': 1, 'pmin': 0.0, 'pmax': 100.0, 'qmin': -50.0, 'qmax': 50.0}
        data['load']['1'] = {'bus': 1, 'pd': 10.0, 'qd': 5.0}
        data['branch']['1'] = {'f_bus': 1, 't_bus': 1, 'br_r': 0.01, 'br_x': 0.1, 'b_fr': 0.0, 'b_to': 0.0, 'g_fr': 0.0,
                               'g_to': 0.0, 'rate_a': 10.0}

    return data


def scale_load(data: Dict[str, Any], scale_coef: List[float]) -> Dict[str, Any]:
    """Scale load data by given coefficients"""
    newdata = copy.deepcopy(data)
    for i, (k, ld) in enumerate(newdata['load'].items()):
        if ld['pd'] > 0:
            ld['pd'] *= scale_coef[i]
            ld['qd'] *= scale_coef[i]
    return newdata


def truncated_normal_samples(mu: float, sigma: float, lower: float, upper: float, n: int) -> List[float]:
    """Generate truncated normal samples"""
    # Use scipy's truncnorm
    a = (lower - mu) / sigma
    b = (upper - mu) / sigma
    samples = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n)
    return samples.tolist()


def get_load_coefficients_fast(mu: float, sigma: float, n: int) -> List[float]:
    """Generate load coefficients with fast method"""
    lower = mu - 0.1
    upper = mu + 0.1
    x = truncated_normal_samples(mu, sigma, lower, upper, n)
    factor = mu * n / sum(x)
    return [max(0.0, min(xi * factor, mu)) for xi in x]


def solve_opf_simplified(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simplified OPF solver using optimization framework
    This is a simplified version - in practice, you'd use proper power flow solvers
    """
    # This is a placeholder for the actual OPF solver
    # In practice, you'd use tools like PYOMO, CVXPY, or call MATPOWER/PowerModels

    # Create a mock solution for demonstration
    solution = {
        'termination_status': 'LOCALLY_SOLVED',
        'objective': np.random.uniform(8000, 12000),
        'solve_time': np.random.uniform(0.1, 2.0),
        'solution': {
            'bus': {},
            'gen': {},
            'branch': {}
        }
    }

    # Generate mock bus solutions
    for bus_id in data['bus']:
        solution['solution']['bus'][bus_id] = {
            'vm': np.random.uniform(0.95, 1.05),
            'va': np.random.uniform(-0.5, 0.5)
        }

    # Generate mock generator solutions
    for gen_id, gen in data['gen'].items():
        if gen['pmax'] > 0:
            solution['solution']['gen'][gen_id] = {
                'pg': np.random.uniform(gen['pmin'], gen['pmax']),
                'qg': np.random.uniform(gen['qmin'], gen['qmax'])
            }

    # Generate mock branch solutions
    for branch_id in data['branch']:
        solution['solution']['branch'][branch_id] = {
            'pf': np.random.uniform(-5, 5),
            'pt': np.random.uniform(-5, 5),
            'qf': np.random.uniform(-3, 3),
            'qt': np.random.uniform(-3, 3)
        }

    return solution


def process_single_run(params: Tuple[int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Process a single OPF run"""
    idx, config = params
    data = config['data']
    Load_range = config['Load_range']
    nperm = config['nperm']

    mu_idx = idx // nperm
    rep = idx % nperm
    mu = Load_range[mu_idx]
    sigma = 0.01  # std dev for load coeffs

    nloads = len(data['load'])
    load_scale = get_load_coefficients_fast(mu, sigma, nloads)
    newdata = scale_load(data, load_scale)

    opf_sol = solve_opf_simplified(newdata)

    if opf_sol['termination_status'] == 'LOCALLY_SOLVED':
        res = {}
        res['scale'] = np.mean(load_scale)
        res['pd'] = {name: load['pd'] for name, load in newdata['load'].items()}
        res['qd'] = {name: load['qd'] for name, load in newdata['load'].items()}

        # Extract generator voltage solutions
        res['vg'] = {}
        for name, gen in newdata['gen'].items():
            if data['gen'][name]['pmax'] > 0:
                bus_id = str(gen['gen_bus'])
                res['vg'][name] = opf_sol['solution']['bus'][bus_id]['vm']

        # Extract generator power solutions
        res['pg'] = {}
        res['qg'] = {}
        for name, gen in opf_sol['solution']['gen'].items():
            if data['gen'][name]['pmax'] > 0:
                res['pg'][name] = gen['pg']
                res['qg'][name] = gen['qg']

        # Extract branch flow solutions
        res['pt'] = {name: branch['pt'] for name, branch in opf_sol['solution']['branch'].items()}
        res['pf'] = {name: branch['pf'] for name, branch in opf_sol['solution']['branch'].items()}
        res['qt'] = {name: branch['qt'] for name, branch in opf_sol['solution']['branch'].items()}
        res['qf'] = {name: branch['qf'] for name, branch in opf_sol['solution']['branch'].items()}

        # Extract bus solutions
        res['va'] = {name: bus['va'] for name, bus in opf_sol['solution']['bus'].items()}
        res['vm'] = {name: bus['vm'] for name, bus in opf_sol['solution']['bus'].items()}

        res['objective'] = opf_sol['objective']
        res['solve_time'] = opf_sol['solve_time']

        return res
    else:
        return None


def main():
    """Main function"""
    args = parse_commandline()

    # Prepare paths
    data_path = Path('data')
    outdir = data_path / 'traindata' / args.netname
    outdir.mkdir(parents=True, exist_ok=True)

    fileout = outdir / f"{args.output}.json"
    filein = data_path / "inputs" / f"{args.netname}.m"

    # Parse input data
    if filein.exists():
        data = parse_matpower_file(str(filein))
    else:
        print(f"Input file not found: {filein}")
        print("Using default 14-bus system structure")
        data = parse_matpower_file(args.netname)

    Load_range = np.arange(args.lb, args.ub + args.step, args.step).tolist()
    nloads = len(data['load'])
    total_runs = len(Load_range) * args.nperm

    print(f"Running with {mp.cpu_count()} CPU cores")
    print(f"Total runs: {total_runs}")

    # Prepare configuration for parallel processing
    config = {
        'data': data,
        'Load_range': Load_range,
        'nperm': args.nperm
    }

    # Run parallel processing
    results = []
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        params = [(i, config) for i in range(total_runs)]

        # Submit all tasks
        future_to_idx = {executor.submit(process_single_run, param): i for i, param in enumerate(params)}

        # Process results with progress bar
        with tqdm(total=total_runs, desc="Processing OPF runs") as pbar:
            for future in as_completed(future_to_idx):
                result = future.result()
                if result is not None:
                    results.append(result)
                pbar.update(1)

    res_stack = [res for res in results if res is not None]

    # Extract constraints
    pglim = {name: (gen['pmin'], gen['pmax']) for name, gen in data['gen'].items() if gen['pmax'] > 0}
    qglim = {name: (gen['qmin'], gen['qmax']) for name, gen in data['gen'].items() if gen['pmax'] > 0}

    vglim = {}
    for name, gen in data['gen'].items():
        if gen['pmax'] > 0:
            bus_id = str(gen['gen_bus'])
            vglim[name] = (data['bus'][bus_id]['vmin'], data['bus'][bus_id]['vmax'])

    vm_lim = {name: (bus['vmin'], bus['vmax']) for name, bus in data['bus'].items()}
    rate_a = {name: branch['rate_a'] for name, branch in data['branch'].items()}

    line_br_rx = {name: (branch['br_r'], branch['br_x']) for name, branch in data['branch'].items()}
    line_bg = {name: (branch['g_to'] + branch['g_fr'], branch['b_to'] + branch['b_fr'])
               for name, branch in data['branch'].items()}

    # Package output
    out_res = {
        'experiments': res_stack,
        'constraints': {
            'vg_lim': vglim,
            'pg_lim': pglim,
            'qg_lim': qglim,
            'vm_lim': vm_lim,
            'rate_a': rate_a,
            'line_rx': line_br_rx,
            'line_bg': line_bg
        }
    }

    # Write JSON output
    with open(fileout, 'w') as f:
        json.dump(out_res, f, indent=4)
        print(f"Saved output to: {fileout}")

    print(f"Successfully processed {len(res_stack)} out of {total_runs} runs")


if __name__ == "__main__":
    main()