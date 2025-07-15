import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from scipy.stats import truncnorm
import pandapower as pp
import pandapower.converter as pc
from concurrent.futures import ThreadPoolExecutor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--netname", "-n", type=str, default="nesta_case14_ieee")
    parser.add_argument("--output", "-o", type=str, default="traindata_ext")
    parser.add_argument("--lb", type=float, default=0.8)
    parser.add_argument("--ub", type=float, default=1.2)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--nperm", type=int, default=10)
    return parser.parse_args()

def scale_load(net, scale_coef):
    new_net = net.deepcopy()
    for i, scale in enumerate(scale_coef):
        if i < new_net.load.shape[0]:
            new_net.load.at[i, 'p_mw'] *= scale
            new_net.load.at[i, 'q_mvar'] *= scale
    return new_net

def truncated_normal_samples(mu, sigma, lower, upper, n):
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    samples = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n)
    # normalize so sum(samples) scaled to mu*n, clamp to [0, mu]
    factor = mu * n / np.sum(samples)
    scaled = np.clip(samples * factor, 0.0, mu)
    return scaled

def run_single_opf(idx, load_range, base_net, nperm, nloads):
    mu_idx = idx // nperm
    mu = load_range[mu_idx]
    sigma = 0.01
    lower = mu - 0.1
    upper = mu + 0.1
    scale_factors = truncated_normal_samples(mu, sigma, lower, upper, nloads)
    new_net = scale_load(base_net, scale_factors)

    try:
        pp.runopp(new_net, verbose=False)
    except Exception as e:
        print(f"OPF failed at idx {idx}, mu={mu:.3f}: {e}")
        return None

    # Check convergence
    if not new_net["converged"]:
        print(f"OPF did not converge at idx {idx}, mu={mu:.3f}")
        return None

    # build and return result dict...
    res = {}
    res["scale"] = np.mean(load_scale)
    res["pd"] = {str(i): float(new_net.load.at[i, 'p_mw']) for i in range(nloads)}
    res["qd"] = {str(i): float(new_net.load.at[i, 'q_mvar']) for i in range(nloads)}
    res["vg"] = {str(i): float(new_net.res_bus.at[bus, 'vm_pu'])
                 for i, bus in enumerate(new_net.gen.bus.values)}
    res["pg"] = {str(i): float(new_net.res_gen.at[i, 'p_mw']) for i in new_net.gen.index}
    res["qg"] = {str(i): float(new_net.res_gen.at[i, 'q_mvar']) for i in new_net.gen.index}
    # branch flows: pt, pf, qt, qf
    res["pt"] = {str(i): float(new_net.res_line.at[i, 'p_to_mw']) for i in new_net.line.index}
    res["pf"] = {str(i): float(new_net.res_line.at[i, 'p_from_mw']) for i in new_net.line.index}
    res["qt"] = {str(i): float(new_net.res_line.at[i, 'q_to_mvar']) for i in new_net.line.index}
    res["qf"] = {str(i): float(new_net.res_line.at[i, 'q_from_mvar']) for i in new_net.line.index}
    # bus voltages: va (angle), vm (magnitude)
    res["va"] = {str(i): float(new_net.res_bus.at[i, 'va_degree']) for i in new_net.bus.index}
    res["vm"] = {str(i): float(new_net.res_bus.at[i, 'vm_pu']) for i in new_net.bus.index}
    res["objective"] = float(new_net.res_cost) if hasattr(new_net, 'res_cost') else None
    # solve time - not directly available in pandapower - omit or estimate
    res["solve_time"] = None
    return res

def main():
    args = parse_args()

    base_path = "/Users/tharuantonymelath/Documents/GitHub/Make_it_work/src/opf-datagen/data"
    outdir = os.path.join(base_path, "traindata", args.netname)
    os.makedirs(outdir, exist_ok=True)

    fileout = os.path.join(outdir, args.output + ".json")
    filein = os.path.join(base_path, "inputs", args.netname + ".m")

    print("Trying to load MPC file from:", filein)
    assert os.path.exists(filein), f"File not found: {filein}"

    base_net = pc.from_mpc(filein, casename_mpc_file=True)

    Load_range = np.arange(args.lb, args.ub + 1e-8, args.step)
    nloads = base_net.load.shape[0]
    total_runs = len(Load_range) * args.nperm

    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_single_opf, i, Load_range, base_net, args.nperm, nloads)
                   for i in range(total_runs)]
        for f in tqdm(futures, total=total_runs):
            res = f.result()
            if res is not None:
                results.append(res)

    # Constraints extraction from base_net (same as Julia code)
    pglim = {str(i): (float(gen["min_p_mw"]), float(gen["max_p_mw"])) for i, gen in base_net.gen.iterrows() if gen["max_p_mw"] > 0}
    qglim = {str(i): (float(gen["min_q_mvar"]), float(gen["max_q_mvar"])) for i, gen in base_net.gen.iterrows() if gen["max_p_mw"] > 0}
    vglim = {str(i): (float(base_net.bus.at[bus, "min_vm_pu"]), float(base_net.bus.at[bus, "max_vm_pu"]))
             for i, bus in enumerate(base_net.gen.bus.values) if base_net.gen.loc[base_net.gen.index[i], "max_p_mw"] > 0}
    vm_lim = {str(i): (float(bus["min_vm_pu"]), float(bus["max_vm_pu"])) for i, bus in base_net.bus.iterrows()}
    rate_a = {str(i): float(branch["max_loading_percent"]) if "max_loading_percent" in branch else 100.0 for i, branch in base_net.line.iterrows()}
    line_br_rx = {str(i): (float(branch["r_ohm_per_km"]), float(branch["x_ohm_per_km"])) for i, branch in base_net.line.iterrows()}
    line_bg = {str(i): (float(branch.get("g_to", 0) + branch.get("g_fr", 0)),
                        float(branch.get("b_to", 0) + branch.get("b_fr", 0))) for i, branch in base_net.line.iterrows()}

    out_res = {
        "experiments": results,
        "constraints": {
            "vg_lim": vglim,
            "pg_lim": pglim,
            "qg_lim": qglim,
            "vm_lim": vm_lim,
            "rate_a": rate_a,
            "line_rx": line_br_rx,
            "line_bg": line_bg
        }
    }

    with open(fileout, "w") as f:
        json.dump(out_res, f, indent=4)

    print("Saved output to:", fileout)

if __name__ == "__main__":
    main()
