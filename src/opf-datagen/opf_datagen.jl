import pandapower as pp
import pandapower.networks as nw
import pandapower.optimal_powerflow as opf

import numpy as np
from scipy.stats import truncnorm
import os
import json
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--netname", "-n", type=str, default="case14")
    parser.add_argument("--output", "-o", type=str, default="traindata_ext")
    parser.add_argument("--lb", type=float, default=0.8)
    parser.add_argument("--ub", type=float, default=1.2)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--nperm", type=int, default=10)
    return parser.parse_args()


def get_network(netname):
    if netname == "case14":
        return nw.case14()
    else:
        raise NotImplementedError(f"Network '{netname}' not supported yet.")


def get_load_coefficients_fast(mu, sigma, n):
    lower, upper = mu - 0.1, mu + 0.1
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    x = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n)
    factor = mu * n / sum(x)
    return np.clip(x * factor, 0.0, mu)


def scale_load(net, scale_factors):
    for i, load in enumerate(net.load.itertuples()):
        net.load.at[load.Index, 'p_mw'] *= scale_factors[i]
        net.load.at[load.Index, 'q_mvar'] *= scale_factors[i]
    return net


def run_single_opf(idx, load_range, base_net, nperm, nloads):
    mu_idx = idx // nperm
    mu = load_range[mu_idx]
    sigma = 0.01

    scale_factors = get_load_coefficients_fast(mu, sigma, nloads)

    net_copy = pp.copy.deepcopy(base_net)
    net_copy = scale_load(net_copy, scale_factors)

    try:
        opf.runopp(net_copy)
        if not net_copy.OPF_converged:
            return None
    except:
        return None

    res = {
        "scale": float(np.mean(scale_factors)),
        "pd": {str(i): float(p) for i, p in enumerate(net_copy.load["p_mw"].values)},
        "qd": {str(i): float(q) for i, q in enumerate(net_copy.load["q_mvar"].values)},
        "vg": {str(i): float(vm) for i, vm in zip(net_copy.gen.index, net_copy.res_bus.loc[net_copy.gen["bus"], "vm_pu"])},
        "pg": {str(i): float(p) for i, p in zip(net_copy.gen.index, net_copy.res_gen["p_mw"].values)},
        "qg": {str(i): float(q) for i, q in zip(net_copy.gen.index, net_copy.res_gen["q_mvar"].values)},
        "vm": {str(i): float(vm) for i, vm in enumerate(net_copy.res_bus["vm_pu"].values)},
        "va": {str(i): float(va) for i, va in enumerate(net_copy.res_bus["va_degree"].values)},
        "objective": float(net_copy.res_cost)
    }
    return res


def main():
    args = parse_args()
    base_net = get_network(args.netname)

    # Output paths
    outdir = os.path.join("C:/dnn-opf/src/data/traindata", args.netname)
    os.makedirs(outdir, exist_ok=True)
    fileout = os.path.join(outdir, args.output + ".json")

    load_range = np.arange(args.lb, args.ub + 1e-8, args.step)
    nloads = len(base_net.load)
    total_runs = len(load_range) * args.nperm

    # Run all OPF scenarios in parallel
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_single_opf, i, load_range, base_net, args.nperm, nloads)
                   for i in range(total_runs)]
        results = [f.result() for f in tqdm(futures)]

    experiments = list(filter(None, results))

    # Gather constraint data
    constraints = {
        "vg_lim": {str(i): (float(row["min_vm_pu"]), float(row["max_vm_pu"]))
                   for i, row in base_net.gen.iterrows()},
        "pg_lim": {str(i): (float(row["min_p_mw"]), float(row["max_p_mw"]))
                   for i, row in base_net.gen.iterrows()},
        "qg_lim": {str(i): (float(row["min_q_mvar"]), float(row["max_q_mvar"]))
                   for i, row in base_net.gen.iterrows()},
        "vm_lim": {str(i): (float(row["min_vm_pu"]), float(row["max_vm_pu"]))
                   for i, row in base_net.bus.iterrows()},
        "rate_a": {str(i): float(row["max_loading_percent"])
                   for i, row in base_net.line.iterrows()},
        "line_rx": {str(i): (float(row["r_ohm_per_km"]), float(row["x_ohm_per_km"]))
                    for i, row in base_net.line.iterrows()}
    }

    out_data = {
        "experiments": experiments,
        "constraints": constraints
    }

    with open(fileout, "w") as f:
        json.dump(out_data, f, indent=4)

    print("Saved output to:", fileout)


if __name__ == "__main__":
    main()
