import re
import subprocess

mO_path = "/usr/local/lib/micromegas/3HDMZ3-2Inert/"
mO_tmp_path = "/usr/local/lib/micromegas/3HDMZ3-2Inert/tmp/"

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
        )
    else:
        mo_output = subprocess.run(
            [mO_path + "main", file_path],
            capture_output=True,
            check=False,
        )
    # 3HDMZ3-2Inert /usr/local/lib/micromegas_6.2.3/
    return mo_output.stdout.decode()


def parse_micromegas_output(text, id_flag=False):
    result = {
        'dark_matter_candidate': {},
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

    if id_flag:
        result |= id_results   # Python 3.9+ (dict union), same as result.update(id_results)


    # Dark Matter Candidate
    dm_match = re.search(
        r"Dark matter candidate is '([^']+)' with spin=.* mass=([\d\.E\+\-]+)",
        text
    )
    if dm_match:
        result['dark_matter_candidate'] = {
            'name': dm_match.group(1),
            'mass': float(dm_match.group(2))
        }
    print(result['dark_matter_candidate'])

    # Odd Particles
    odd_particles_match = re.search(
        r"Masses of odd sector Particles:(.*?)(?:====|$)",
        text, re.S
    )
    if odd_particles_match:
        odd_section = odd_particles_match.group(1)
        odd_matches = re.findall(
            r"([~A-Za-z0-9]+)\s*:\s*m\w+\s*=\s*([\d\.E\+\-]+)",
            odd_section
        )
        for name, mass in odd_matches:
            result['odd_particles'].append({'name': name, 'mass': float(mass)})

    print(result['odd_particles'])

    # Relic Density
    relic_match = re.search(r"Xf=.*Omega=([\d\.Ee\+\-]+)", text)
    if relic_match:
        result['relic_density']['Omega'] = float(relic_match.group(1))

    print(result['relic_density']['Omega'])
    # Indirect Detection - Annihilation Cross Section
    if id_flag:
        ann_cs_match = re.search(r'annihilation cross section\s+([0-9.Ee+-]+)\s*cm\^3/s', text)
        if ann_cs_match:
            result['indirect_detection']['annihilation_cross_section'] = float(ann_cs_match.group(1))

        # Indirect Detection - dominant processes for H1 and A1
        process_matches = re.findall(r"([~A-Za-z0-9, ]+)->([~A-Za-z0-9+\- ]+)\s+([\d\.E\+\-]+)", text)
        process_dict = {}

        for proc_in, proc_out, val in process_matches:
            in_parts = [p.strip() for p in proc_in.split(',')]
            out_parts = [p.strip() for p in proc_out.split(',')]
            
            if len(in_parts) == 2:
                key = f"{in_parts[0]}{in_parts[1]}"
                val = float(val)
                
                # Save the whole process: (incoming particles, outgoing particles) : value
                if key not in process_dict or process_dict[key][1] < val:
                    process_dict[key] = ((f"{in_parts[0]}{in_parts[1]}", ''.join(out_parts)), val)

        # Extract dominant processes for ~H1~H1 and ~~A1~~A1
        dominant = []
        for key in ['~H1~H1', '~~A1~~A1']:
            if key in process_dict:
                # Save tuple of (incoming, outgoing) and its value
                dominant.append((process_dict[key][0], process_dict[key][1]))

        result['indirect_detection']['dominant_processes'] = dominant


        # Photon, Positron, Antiproton flux
        flux_match = re.search(r"Photon flux =\s*([\d\.E\+\-]+).*?\nPositron flux\s*=\s*([\d\.E\+\-]+).*?\nAntiproton flux\s*=\s*([\d\.E\+\-]+)", text, re.S)
        if flux_match:
            result['photon_flux'] = float(flux_match.group(1))
            result['positron_flux'] = float(flux_match.group(2))
            result['antiproton_flux'] = float(flux_match.group(3))


    # Direct Detection
    dd_match = re.search(
        r"([~A-Za-z0-9\[\]]+)-nucleon micrOMEGAs amplitudes.*?"
        r"proton:.*?SI\s+([-\d\.E\+\-]+).*?"
        r"neutron: SI\s+([-\d\.E\+\-]+).*?\n"
        r".*?cross sections\[pb\]:\s*proton\s*SI\s*([\d\.E\+\-]+).*?"
        r"neutron\s*SI\s*([\d\.E\+\-]+)",
        text, re.S
    )
    if dd_match:
        dm_clean = re.sub(r'[\[\]]', '', dd_match.group(1))
        result['direct_detection'][dm_clean] = {
            'SI_proton': float(dd_match.group(2)),
            'SI_neutron': float(dd_match.group(3)),
            'CS_proton': float(dd_match.group(4)),
            'CS_neutron': float(dd_match.group(5))
        }

    # Exclusion
    excl_match = re.search(r"Excluded by ([A-Za-z0-9_]+)\s+([\d\.]+)%", text)
    if excl_match:
        result['exclusion'] = {
            'experiment': excl_match.group(1),
            'exclusion_percent': float(excl_match.group(2))
        }

    return result



def processMOml(output, dd_flag=False, id_flag=False):
    result = {}

    relic = output.get('relic_density', {})
    if dd_flag:
        direct = output.get('direct_detection', {})
    if id_flag:
        indirect = output.get('indirect_detection', {})

    # Relic densities
    result['Omega_1'] = relic.get('Omega', None)
    result['Omega_2'] = 0.0 # relic.get('Omega', None)
    # relic.get('Omega_2h2', None)

    if dd_flag:
        # Direct detection cross sections (example for ~H1 and ~~A1)
        result['dd_H1_SI_CS'] = direct.get('~H1~H1', {}).get('CS_proton', None)
        result['dd_A1_SI_CS'] = direct.get('~H1~H1', {}).get('CS_proton', None)
        # direct.get('~~A1~~A1', {}).get('CS_proton', None)

    if id_flag:
        # Indirect detection annihilation cross-section
        result['id_ann_CS'] = indirect.get('annihilation_cross_section', None)

        # Indirect detection dominant processes
        dom_procs = indirect.get('dominant_processes', [])

        # Handle H1 and A1 dominant processes if available
        if len(dom_procs) > 0:
            result['id_H1_dom'] = dom_procs[0][0][1]  # outgoing particles
            result['id_H1_dom_perc'] = dom_procs[0][1]  # percentage
        else:
            result['id_H1_dom'] = None
            result['id_H1_dom_perc'] = None

        if len(dom_procs) > 1:
            result['id_A1_dom'] = dom_procs[1][0][1]
            result['id_A1_dom_perc'] = dom_procs[1][1]
        else:
            result['id_A1_dom'] = None
            result['id_A1_dom_perc'] = None

        # Fluxes
        result['id_photon_flux'] = output.get('photon_flux', None)
        result['id_positron_flux'] = output.get('positron_flux', None)
        result['id_antiproton_flux'] = output.get('antiproton_flux', None)

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