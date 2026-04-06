import re
import subprocess
import pandas as pd
import numpy as np


mO_path = "/usr/local/lib/micromegas/3HDMZ3-2Inert/"
mO_tmp_path = "/usr/local/lib/micromegas/3HDMZ3-2Inert/tmp/"
# mO_path = "/home/figueiredo/software/micromegas_6.2.3/3HDMZ3-2Inert/"


def run_micromegas(file_path, mO_path, dd_flag=False, id_flag=False):
    # mo_output = subprocess.run(
    #     ["/home/figueiredo/software/micromegas_6.2.3/3HDMZ3-2Inert/main", file_path],
    #     capture_output=True,
    #     check=False,
    # )
    if id_flag:
        mo_output = subprocess.run(
            [mO_path + "main_id", file_path],
            capture_output=True,
            check=False,
            timeout=300,
        )
    else:
        mo_output = subprocess.run(
            [mO_path + "main", file_path],
            capture_output=True,
            check=False,
        )
    # 3HDMZ3-2Inert /usr/local/lib/micromegas_6.2.3/
    return mo_output.stdout.decode()


# def parse_and_log(df, pattern, text, key, min_value=None):
#     match = re.search(pattern, text)
#     if match:
#         try:
#             value = float(match.group(1))
#             if min_value is not None and value < min_value:
#                 raise ValueError(f"{key} below minimum value {min_value}")
#             df['relic_density'][key] = value
#             print(f"{key} parsed: {value}", flush=True)
#         except ValueError:
#             df['relic_density'][key] = 0.0
#             print(f"{key} invalid value, set to 0.0", flush=True)
#     else:
#         df['relic_density'][key] = None
#         print(f"{key} not parsed: {df['relic_density'][key]}", flush=True)


# works but stores as list
def parse_micromegas_output(text,  dd_flag=False, id_flag=False, ch_relic_flag=False):
    result = {
        'dark_matter_candidates': [],
        'odd_particles': [],
        'relic_density': {},
        'direct_detection': {},
        'exclusion': {}
    }

    id_results = {        
        'indirect_detection': {'annihilation_cross_section': None, 'dominant_processes': []},
        'photon_flux': None,
        'positron_flux': None,
        'antiproton_flux': None,
    }

    print(text)

    # if id_flag:
    result |= id_results   # Python 3.9+ (dict union), same as result.update(id_results)

    # Dark Matter Candidates
    dm_matches = re.findall(r"Dark matter candidate is '([^']+)' with spin=.* mass=([\d\.E\+\-]+)", text)
    for name, mass in dm_matches:
        result['dark_matter_candidates'].append({'name': name, 'mass': float(mass)})

    print(dm_matches)

    # Odd Particles
    odd_particles_match = re.search(r"Masses of odd sector Particles:(.*?)(?:====|$)", text, re.S)
    if odd_particles_match:
        odd_section = odd_particles_match.group(1)
        odd_matches = re.findall(r"([~A-Za-z0-9]+)\s*:\s*m\w+\s*=\s*([\d\.E\+\-]+)", odd_section)
        for name, mass in odd_matches:
            result['odd_particles'].append({'name': name, 'mass': float(mass)})

    print(odd_matches)

    # Relic Density
    # relic_match = re.search(r"Omega_1h\^2=([\d\.E\+\-]+)\s+Omega_2h\^2=([\d\.E\+\-]+)", text)
    # if relic_match:
    #     result['relic_density']['Omega_1h2'] = float(relic_match.group(1))
    #     result['relic_density']['Omega_2h2'] = float(relic_match.group(2))
    #     result['relic_density']['OmegaT'] = result['relic_density']['Omega_1h2'] + result['relic_density']['Omega_2h2']

    omega_match = re.search(r"Omega\s*=\s*([\d\.E\+\-]+)", text)
    # OmegaT = float(omega_match.group(1))
    print(omega_match)

    def safe_float(match):
        if not match:
            return None
        try:
            value = float(match.group(1))
            if np.isfinite(value):
                return value
        except ValueError:
            pass
        return None

    OmegaT = safe_float(omega_match)

    # # Omega total
    # omega_match = re.search(r"Omega\s*=\s*([\d\.E\+\-]+)", text)
    # OmegaT = float(omega_match.group(1))
    # print("omega_matches")
    # print(omega_match)

    omega1_match = re.search(r"Omega_1\s*=\s*([\d\.E\+\-]+)", text)
    # Omega_1 = float(omega1_match.group(1))
    # print(omega1_match)
    Omega_1 = safe_float(omega1_match)

    omega2_match = re.search(r"Omega_2\s*=\s*([\d\.E\+\-]+)", text)
    # Omega_2 = float(omega2_match.group(1))
    # print(omega2_match)
    Omega_2 = safe_float(omega2_match)


    # # Helper function to handle non-negative values or None
    def parse_omega_k(match, value):
        if match:
            return max(value, 0.0)
        return None

    def parse_omega(match, value):
        if not match:
            return None
        if value is None or value <= 0.0:
            return None     # treat invalid values as "not parsed"
        return value


    # # Omega_T
    # if omega_match:
    #     Omega_T = parse_omega(omega_match, OmegaT)
    #     result['relic_density']['OmegaT'] = Omega_T

    #     # Omega_1
    #     result['relic_density']['Omega_1h2'] = parse_omega(omega1_match, Omega_1)
    #     if omega1_match is False:
    #         print(f"Omega1 not parsed: None", flush=True)

    #     # Omega_2
    #     Omega_2h2 = parse_omega(omega2_match, Omega_2)
    #     result['relic_density']['Omega_2h2'] = Omega_2h2
    #     if omega2_match is False:
    #         print(f"Omega2 not parsed: None", flush=True)

    #     # Adjust Omega_T if Omega_2 is negative
    #     if omega2_match and Omega_2 < 0.0:
    #         result['relic_density']['OmegaT'] = result['relic_density']['Omega_1h2']

    # else:
    #     result['relic_density']['OmegaT'] = None
    #     result['relic_density']['Omega_1h2'] = None
    #     result['relic_density']['Omega_2h2'] = None
    #     print(f"OmegaT not parsed: None", flush=True)


    # Normalize Omega_T first
    Omega_T = parse_omega(omega_match, OmegaT)

    if Omega_T is None:
        # Parsing failed or invalid → set all relic density values to None
        result['relic_density']['OmegaT'] = None
        result['relic_density']['Omega_1h2'] = None
        result['relic_density']['Omega_2h2'] = None
        print("OmegaT not parsed: None", flush=True)
    else:
        # Parsing succeeded → fill the others
        result['relic_density']['OmegaT'] = Omega_T

        # Omega_1
        Omega_1h2 = parse_omega_k(omega1_match, Omega_1)
        result['relic_density']['Omega_1h2'] = Omega_1h2
        if omega1_match is False:
            print("Omega1 not parsed: None", flush=True)

        # Omega_2
        Omega_2h2 = parse_omega_k(omega2_match, Omega_2)
        result['relic_density']['Omega_2h2'] = Omega_2h2
        if omega2_match is False:
            print("Omega2 not parsed: None", flush=True)

        # Adjust Omega_T if Omega_2 is negative
        if omega2_match and Omega_2 is not None and Omega_2 < 0.0:
            result['relic_density']['OmegaT'] = Omega_1h2


        if ch_relic_flag:
            # Capture the entire list of relic-density channels
            channels_relic = []

            channel_matches = re.findall(
                r"([0-9]+\.[0-9]+)%\s+([~A-Za-z0-9_+-]+)\s+([~A-Za-z0-9_+-]+)\s*->\s*([~A-Za-z0-9_+\- ]+)",
                text
            )

            for pct, p1, p2, out in channel_matches:
                channels_relic.append({
                    "percentage": float(pct),
                    "incoming": [p1, p2],
                    "outgoing": [p.strip() for p in out.split()],
                })

            result["channels_relic"] = channels_relic

    # # Indirect Detection - Annihilation Cross Section
    # if id_flag:
    #     ann_cs_match = re.search(r'annihilation cross section\s+([0-9.Ee+-]+)\s*cm\^3/s', text)
    #     if ann_cs_match:
    #         result['indirect_detection']['annihilation_cross_section'] = float(ann_cs_match.group(1))

    #     # Indirect Detection - dominant processes for H1 and A1
    #     process_matches = re.findall(r"([~A-Za-z0-9, ]+)->([~A-Za-z0-9+\- ]+)\s+([\d\.E\+\-]+)", text)
    #     process_dict = {}

    #     for proc_in, proc_out, val in process_matches:
    #         in_parts = [p.strip() for p in proc_in.split(',')]
    #         out_parts = [p.strip() for p in proc_out.split(',')]
            
    #         if len(in_parts) == 2:
    #             key = f"{in_parts[0]}{in_parts[1]}"
    #             val = float(val)
                
    #             # Save the whole process: (incoming particles, outgoing particles) : value
    #             if key not in process_dict or process_dict[key][1] < val:
    #                 process_dict[key] = ((f"{in_parts[0]}{in_parts[1]}", ''.join(out_parts)), val)

    #     # Extract dominant processes for ~H1~H1 and ~~A1~~A1
    #     dominant = []
    #     for key in ['~H1~H1', '~~A1~~A1']:
    #         if key in process_dict:
    #             # Save tuple of (incoming, outgoing) and its value
    #             dominant.append((process_dict[key][0], process_dict[key][1]))

    #     result['indirect_detection']['dominant_processes'] = dominant

    # Indirect Detection - Annihilation Cross Section
    if id_flag:
        ann_cs_match = re.search(r'annihilation cross section\s+([0-9.Ee+-]+)\s*cm\^3/s', text)
        if ann_cs_match:
            result['indirect_detection']['annihilation_cross_section'] = float(ann_cs_match.group(1))

        # Extract ALL processes with incoming ~H1,~H1 or ~~A1,~~A1
        process_matches = re.findall(
            r"([~A-Za-z0-9, ]+)->([~A-Za-z0-9+\- ]+)\s+([\d\.E\+\-]+)",
            text
        )

        selected_processes = []

        for proc_in, proc_out, val in process_matches:
            in_parts = [p.strip() for p in proc_in.split(',')]

            # # Keep ALL (H1 H1) and (A1 A1) processes
            # if len(in_parts) == 2 and (
            #     (in_parts[0] == "~H1" and in_parts[1] == "~H1") or
            #     (in_parts[0] == "~~A1" and in_parts[1] == "~~A1")
            # ):
            if len(in_parts) == 2:
                selected_processes.append({
                    "incoming": in_parts,
                    "outgoing": [p.strip() for p in proc_out.split(',')],
                    "value": float(val)
                })

        result['indirect_detection']['selected_processes'] = selected_processes


        # Photon, Positron, Antiproton flux
        flux_match = re.search(r"Photon flux =\s*([\d\.E\+\-]+).*?\nPositron flux\s*=\s*([\d\.E\+\-]+).*?\nAntiproton flux\s*=\s*([\d\.E\+\-]+)", text, re.S)
        if flux_match:
            result['photon_flux'] = float(flux_match.group(1))
            result['positron_flux'] = float(flux_match.group(2))
            result['antiproton_flux'] = float(flux_match.group(3))

    # Direct Detection
    dd_matches = re.findall(r"([~A-Za-z0-9\[\]]+)-nucleon micrOMEGAs amplitudes.*?proton:.*?SI\s+([-\d\.E\+\-]+).*?neutron: SI\s+([-\d\.E\+\-]+).*?\n.*?cross sections\[pb\]:\s*proton\s*SI\s*([\d\.E\+\-]+).*?neutron\s*SI\s*([\d\.E\+\-]+)", text, re.S)
    print(dd_matches)
    for dm, si_p, si_n, cs_p, cs_n in dd_matches:
        dm_clean = re.sub(r'[\[\]]', '', dm)  # remove brackets
        result['direct_detection'][dm_clean] = {
            'SI_proton': float(si_p),
            'SI_neutron': float(si_n),
            'CS_proton': float(cs_p),   
            'CS_neutron': float(cs_n)
        }
    # dd_matches = re.findall(
    #     r"([~A-Za-z0-9\[\]]+)-nucleon micrOMEGAs amplitudes.*?"
    #     r"proton:\s*SI\s+([-\d\.E\+\-]+).*?SD\s+([-\d\.E\+\-]+).*?"
    #     r"neutron:\s*SI\s+([-\d\.E\+\-]+).*?SD\s+([-\d\.E\+\-]+)",
    #     text, re.S
    # )

    # for dm, si_p, sd_p, si_n, sd_n in dd_matches:
    #     dm_clean = re.sub(r'[\[\]]', '', dm)  # remove brackets
    #     result['direct_detection'][dm_clean] = {
    #         'SI_proton_amp': float(si_p),
    #         'SD_proton_amp': float(sd_p),
    #         'SI_neutron_amp': float(si_n),
    #         'SD_neutron_amp': float(sd_n)
    #     }

    # Exclusion
    excl_match = re.search(r"Excluded by ([A-Za-z0-9_]+)\s+([\d\.]+)%", text)
    if excl_match:
        result['exclusion'] = {
            'experiment': excl_match.group(1),
            'exclusion_percent': float(excl_match.group(2))
        }

    # Higgs invisible branching ratios
    higgs_matches = re.findall(r"([\d\.E\+\-]+)\s+h\s*->\s*([~A-Za-z0-9, ]+)", text)
    if higgs_matches:
        result['higgs_branchings'] = []
        for br, channel in higgs_matches:
            br_val = float(br)
            channel_clean = channel.replace(" ", "")
            result['higgs_branchings'].append({'channel': channel_clean, 'BR': br_val})

        # Look for invisible channels explicitly
        for hb in result['higgs_branchings']:
            if hb['channel'] == "~H1,~H1":
                result['BR_h_to_H1H1'] = hb['BR']
            if hb['channel'] == "~~H2,~~H2":
                result['BR_h_to_H2H2'] = hb['BR']

    # Higgs total width
    higgs_width_match = re.search(r"h\s*:\s*total width\s*=\s*([\d\.E\+\-]+)", text)
    # print("h width")
    # print(higgs_width_match)
    if higgs_width_match:
        result['Gamma_h'] = float(higgs_width_match.group(1))

    H1_width_match = re.search(r"~H1\s*:\s*total width\s*=\s*([\d\.E\+\-]+)", text)
    # print("h width")
    # print(H1_width_match)
    if H1_width_match:
        result['Gamma_H1'] = float(H1_width_match.group(1))

    A1_width_match = re.search(r"~A1\s*:\s*total width\s*=\s*([\d\.E\+\-]+)", text)
    # print("h width")
    # print(H2_width_match)
    if A1_width_match:
        result['Gamma_A1'] = float(A1_width_match.group(1))

    A2_width_match = re.search(r"~~A2\s*:\s*total width\s*=\s*([\d\.E\+\-]+)", text)
    # print("h width")
    # print(H2_width_match)
    if A2_width_match:
        result['Gamma_A2'] = float(A2_width_match.group(1))

    H2_width_match = re.search(r"~~H2\s*:\s*total width\s*=\s*([\d\.E\+\-]+)", text)
    # print("h width")
    # print(H2_width_match)
    if H2_width_match:
        result['Gamma_H2'] = float(H2_width_match.group(1))

    return result




def processMOml(output, dd_flag=False, id_flag=False, ch_relic_flag=False):
    result = {}

    relic = output.get('relic_density', {})
    # if dd_flag:
    dm_candidates = output.get('dark_matter_candidates')
    direct = output.get('direct_detection', {})
    # if id_flag:
    indirect = output.get('indirect_detection', {})

    # Relic densities
    # result['Omega_1'] = relic.get('Omega_1h2', None)
    # result['Omega_2'] = relic.get('Omega_2h2', None)

    Omega1 = relic.get('Omega_1h2', None)
    Omega2 = relic.get('Omega_2h2', None)
    OmegaT = relic.get('OmegaT', None)
    print(f"OmegaT: {OmegaT}")


    result['Omega_1'] = Omega1
    result['Omega_2'] = Omega2
    result['OmegaT'] = OmegaT

    # Omega_T = None
    # if Omega1 is not None and Omega2 is not None:
    #     Omega_T = Omega1 + Omega2

    # if dd_flag:
    # Direct detection cross sections (example for ~H1 and ~~A1)
    # dd_H1_CS = direct.get('~H1~H1', {}).get('CS_proton', None) + direct.get('~H1~H1', {}).get('CS_neutron', None)
    # dd_A1_CS = direct.get('~~A1~~A1', {}).get('CS_proton', None) + direct.get('~~A1~~A1', {}).get('CS_neutron', None)
    # result['dd_H1_SI_CS'] = dd_H1_CS
    # result['dd_A1_SI_CS'] = dd_A1_CS

            # Constants for Xenon
    if OmegaT is not None:
        Z = 54
        A = 131
        m_N = A * 0.938  # approx nucleus mass
        gev2_to_pb = 3.89379e8

        # Helper function
        def calc_sigma(mass, fp_mO, fn_mO, Omega_k, Omega_T):
            m_dm = mass
            fp = fp_mO
            fn = fn_mO

            print(f"m_dm = {m_dm}, fp = {fp}, fn = {fn}, Omega_k = {Omega_k}, Omega_T = {Omega_T}")
            mu_k = (m_dm * m_N) / (m_dm + m_N)
            sigma_SI_Xe = (4 * mu_k**2 / (np.pi * A**2)) * (Z*fp + (A-Z)*fn)**2
            xi_k = (Omega_k / Omega_T)
            sigma_SI_r = sigma_SI_Xe * xi_k

            return sigma_SI_r * gev2_to_pb

        mass_map = {d['name']: d['mass'] for d in dm_candidates}
        mH1 = mass_map.get('~H1')
        mA1 = mass_map.get('~~H2')

        fp_H1_SI = direct.get('~H1~H1', {}).get('SI_proton', None)
        fn_H1_SI = direct.get('~H1~H1', {}).get('SI_neutron', None)
        fp_A1_SI = direct.get('~~H2~~H2', {}).get('SI_proton', None)
        print(fp_A1_SI)
        fn_A1_SI = direct.get('~~H2~~H2', {}).get('SI_neutron', None)
        print(fn_A1_SI)


        sigma_H1 = calc_sigma(mH1, fp_H1_SI, fn_H1_SI, result['Omega_1'], result['OmegaT'])
        sigma_A1 = calc_sigma(mA1, fp_A1_SI, fn_A1_SI, result['Omega_2'], result['OmegaT'])

        result['dd_H1_SI_CS'] = sigma_H1
        result['dd_H2_SI_CS'] = sigma_A1

    else:
        result['dd_H1_SI_CS'] = None
        result['dd_H2_SI_CS'] = None

    if id_flag:
    # # Indirect detection annihilation cross-section
        result['id_ann_CS'] = indirect.get('annihilation_cross_section', None)

    # # Indirect detection dominant processes
    # dom_procs = indirect.get('dominant_processes', [])

    # # Handle H1 and A1 dominant processes if available
    # if len(dom_procs) > 0:
    #     result['id_H1_dom'] = dom_procs[0][0][1]  # outgoing particles
    #     result['id_H1_dom_perc'] = dom_procs[0][1]  # percentage
    # else:
    #     result['id_H1_dom'] = None
    #     result['id_H1_dom_perc'] = None

    # if len(dom_procs) > 1:
    #     result['id_A1_dom'] = dom_procs[1][0][1]
    #     result['id_A1_dom_perc'] = dom_procs[1][1]
    # else:
    #     result['id_A1_dom'] = None
    #     result['id_A1_dom_perc'] = None

    # --- Indirect Detection: Dominant Channels ---

    selected_procs = indirect.get("selected_processes", [])

    # Initialize result fields
    result["id_H1_dom"] = None
    result["id_H1_dom_perc"] = None
    result["id_H2_dom"] = None
    result["id_H2_dom_perc"] = None
    result["id_dom_channel"] = None
    result["id_dom_value"] = None

    if selected_procs:
        from collections import defaultdict

        # Total contributions by outgoing state (all channels, including cross)
        total_contrib = defaultdict(float)
        for proc in selected_procs:
            out_state = " ".join(proc["outgoing"])
            total_contrib[out_state] += proc["value"]

        # WW and ZZ totals (including cross channels)
        WW_total = sum(val for state, val in total_contrib.items() if "WP" in state or "W+" in state or "W-" in state)
        ZZ_total = sum(val for state, val in total_contrib.items() if state.strip() == "Z Z")

        # Total electroweak signal
        EW_total = WW_total + ZZ_total

        # Determine overall dominant channel
        max_other_state_val = max([v for s, v in total_contrib.items() 
                                if s.strip() not in ["WP WP~", "Z Z"]], default=0.0)

        if EW_total >= max_other_state_val:
            result["id_dom_channel"] = "WW+ZZ"
            result["id_dom_value"] = EW_total
        else:
            # Dominant non-EW channel
            dom_state = max(total_contrib, key=total_contrib.get)
            result["id_dom_channel"] = dom_state
            result["id_dom_value"] = total_contrib[dom_state]

        # --- Per-DM dominant channels (H1 and A1) ---
        # Only include pure DM channels, ignore cross channels
        H1_contrib = defaultdict(float)
        A1_contrib = defaultdict(float)

        for proc in selected_procs:
            incoming = proc["incoming"]
            val = proc["value"]
            out_state = " ".join(proc["outgoing"])

            # H1 only
            if incoming == ["~H1", "~H1"]:
                H1_contrib[out_state] += val
            # A1 only
            elif incoming == ["~~H2", "~~H2"]:
                A1_contrib[out_state] += val

        # Pick dominant H1
        if H1_contrib:
            dom_H1_state = max(H1_contrib, key=H1_contrib.get)
            result["id_H1_dom"] = dom_H1_state
            result["id_H1_dom_perc"] = H1_contrib[dom_H1_state]

        # Pick dominant A1
        if A1_contrib:
            dom_A1_state = max(A1_contrib, key=A1_contrib.get)
            result["id_H2_dom"] = dom_A1_state
            result["id_H2_dom_perc"] = A1_contrib[dom_A1_state]

    

    # Fluxes
    result['id_photon_flux'] = output.get('photon_flux', None)
    result['id_positron_flux'] = output.get('positron_flux', None)
    result['id_antiproton_flux'] = output.get('antiproton_flux', None)

    # Higgs invisible branching ratios
    result['BR_h_to_H1H1'] = output.get('BR_h_to_H1H1', None)
    result['BR_h_to_H2H2'] = output.get('BR_h_to_H2H2', None)

    result['Gamma_h'] = output.get('Gamma_h', None)
    result['Gamma_H1'] = output.get('Gamma_H1', None)
    result['Gamma_A1'] = output.get('Gamma_A1', None)
    result['Gamma_H2'] = output.get('Gamma_H2', None)
    result['Gamma_A2'] = output.get('Gamma_A2', None)

    return result

def write_mO_parameters_file(results, filepath):
    L1 = results["L1"].values[0]
    L2 = results["L2"].values[0]
    L4 = results["L4"].values[0]
    L7 = results["L7"].values[0]
    L10 = results["L10"].values[0]
    L11 = results["L11"].values[0]
    theta = results["theta"].values[0]
    g1 = results["g1"].values[0]
    g2 = results["g2"].values[0]
    mH1 = results["MH10"].values[0]
    mH2 = results["MH20"].values[0]
    mA1 = results["MH10"].values[0]
    mA2 = results["MH20"].values[0]
    mH1P = results["mC1"].values[0]
    mH2P = results["mC2"].values[0]
    content = f"""# file to improve default parameters
        L1      {L1}
        L2      {L2}
        L4      {L4}
        L7      {L7}
        L10     {L10}
        L11     {L11}
        theta   {theta}
        g1      {g1}
        g2      {g2}
        mH1     {mH1}
        mH2     {mH2}
        mA1     {mA1}
        mA2     {mA2}
        mH1P    {mH1P}
        mH2P    {mH2P}"""

    
    with open(filepath, 'w') as f:
        f.write(content)


# Example usage:
# output = run_micromegas("/home/figueiredo/software/micromegas_6.2.3/3HDMZ3-2Inert/data.par")
# parsed_results = parse_micromegas_output(output)

# import pprint
# pprint.pprint(parsed_results)


# results = processMOml(parsed_results)
# print("Processed Results:")
# print(results)