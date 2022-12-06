import datetime
import os

def build_mk_script_file(settings_loaded,dir_path):
    name = settings_loaded['system']['name']
    md_engine = settings_loaded['system']['path_to_md_engine']
    topo = os.path.relpath(settings_loaded['system']['topo_file'], dir_path)
    md_dir_name = os.path.basename(os.path.normpath(settings_loaded['system']['md_dir']))
    cnf = os.path.relpath(f"{md_dir_name}/{settings_loaded['system']['cnf_file']}",  dir_path)
    pert =  os.path.relpath(settings_loaded['system']['pert_file'], dir_path)
    version = settings_loaded['system']['version']
    if settings_loaded['system']['lib_type'] == f"cuda":
        lib = f'mk_script_cuda_8_slurm.lib'
    elif settings_loaded['system']['lib_type'] == f"cuda_local":
        lib=f'mk_script_cuda_8.lib'
    body = f"""@sys            aeds_{name}
@bin           {md_engine}
@dir           {dir_path}
@files
  topo          {topo}
  input         aeds.imd
  coord        ../../../{md_dir_name}/{cnf}
  pttopo       {pert}
@template       {lib}
@version        {version}
@joblist        aeds.job"""
    return body

def build_job_file(settings_loaded):
    if settings_loaded['simulation']['NSTATS']>1:
        body = f"""TITLE
splitted search in {settings_loaded['simulation']['parameters']['NRUN']} parts
END
JOBSCRIPTS 
job_id NTIVEL NTIAEDSS subdir run_after
"""
        for i in range(1,settings_loaded['simulation']['parameters']['NRUN']+1,1):
            if i == 1:
                body += f"{i}     1      1        .      {i-1}\n"
            elif i < settings_loaded['simulation']['parameters']['NRUN']:
                body += f"{i}     0      0        .      {i-1}\n"
            elif i == settings_loaded['simulation']['parameters']['NRUN']:
                body += f"{i}     0      0        .      {i-1}\n"
                body += f"END"
        return body
    else: 
        body = f"""TITLE
splitted search in {settings_loaded['simulation']['parameters']['NRUN']} parts
END
JOBSCRIPTS 
job_id NTIAEDSS subdir run_after
"""
        for i in range(1,settings_loaded['simulation']['parameters']['NRUN']+1,1):
            if i == 1:
                body += f"{i}         1        .      {i-1}\n"
            elif i < settings_loaded['simulation']['parameters']['NRUN']:
                body += f"{i}         0        .      {i-1}\n"
            elif i == settings_loaded['simulation']['parameters']['NRUN']:
                body += f"{i}         0        .      {i-1}\n"
                body += f"END"
        return body

def scrap_ref_imd(settings_loaded):
    ref_imd_path = f"{settings_loaded['system']['md_dir']}/{settings_loaded['system']['ref_imd']}"
    flag = 0
    temp_flag = 0
    body_1 =f""
    body_2 =f""
    with open(ref_imd_path, 'r') as file:
        for line in file:
            if f"SYSTEM" in line:
                flag = 1
            elif f"STEP" in line:
                flag = 0
            if flag == 1:
                body_1 += f"{line}"
            if f"BOUNDCOND" in line:
                flag = 2
            elif f"INITIALISE" in line:
                flag = 0
            if flag == 2:
                body_2 += f"{line}"
            if temp_flag == 3:
                temp_flag = 0
                settings_loaded['simulation']['parameters']['temp'] = int(line.split()[0])
            if f"TEMP0(1 ... NBATHS)" in line:
                temp_flag = 3
    return body_1,body_2
                    
def build_imd_file(settings_loaded,EIR,rs):
    date = datetime.date.today()
    NSTATES = settings_loaded['simulation']['NSTATES']
    NSTLIM = settings_loaded['simulation']['parameters']['NSTLIM']
    NTPR = settings_loaded['simulation']['parameters']['NTPR']
    NTWX = settings_loaded['simulation']['parameters']['NTWX']
    NTWE = settings_loaded['simulation']['parameters']['NTWE']
    dt = settings_loaded['simulation']['parameters']['dt']
    EMIN = settings_loaded['simulation']['parameters']['EMIN']
    EMAX = settings_loaded['simulation']['parameters']['EMAX']
    if settings_loaded['system']['lib_type'] == f"cuda":
        ALPHLJ='0'  
        ALPHCRF='0'
    elif settings_loaded['system']['lib_type'] == f"cuda_local":
        ALPHLJ=''  
        ALPHCRF=''
    rnd_seed = 210184 + rs
    b1,b2=scrap_ref_imd(settings_loaded)
    body = f"""TITLE
	Automatically generated input file
	bbraun {date}
END
{b1}STEP
#   NSTLIM         T        DT
   {NSTLIM}        0       {dt}
END
{b2}
INITIALISE
# Default values for NTI values: 0
#   NTIVEL    NTISHK    NTINHT    NTINHB
         0         0         0         0
#   NTISHI    NTIRTC    NTICOM
         1         0         0
#   NTISTI
         0
#       IG     TEMPI
  {rnd_seed}     300
END
COMTRANSROT
#     NSCM
      1000
END
PRINTOUT
#NTPR: print out energies, etc. every NTPR steps
#NTPP: =1 perform dihedral angle transition monitoring
#     NTPR      NTPP
     {NTPR}        0
END
WRITETRAJ
#    NTWX     NTWSE      NTWV      NTWF      NTWE      NTWG      NTWB
    {NTWX}        0         0         0     {NTWE}        0         0
END
AEDS
#     AEDS
         1
#   ALPHLJ   ALPHCRF      FORM      NUMSTATES
    {ALPHLJ} {ALPHCRF}       1      {NSTATES}
#     EMAX      EMIN
    {EMAX}    {EMIN}
# EIR [1..NUMSTATES]
         0      {EIR}
# NTIAEDSS  RESTREMIN  BMAXTYPE      BMAX    ASTEPS    BSTEPS
         1          1         2         2       500     10000
END"""

    if settings_loaded['system']['lib_type'] == f"cuda" or settings_loaded['system']['lib_type'] == f"cuda_local":
        body += F"""
INNERLOOP
#     NTILM      NTILS      NGPUS      NDEVG
         4         0         1         0
END"""
    return body

def build_ene_ana(settings_loaded,NRUN):
    name = settings_loaded['system']['name']
    topo = settings_loaded['system']['topo_file']
    equilibrate= settings_loaded['simulation']['equilibrate']
    if equilibrate[0] == True:
        start = NRUN*(equilibrate[1]/100)
        start = round(start)
    else:
        start = 1
    NRUN = NRUN + 1

    body = f"""@prop eds_vr e1 e2 e1s e2s e1r e2r eds_emin eds_emax eds_vmix eds_globmin eds_globminfluc
@topo ../../../../topo/{topo}
@library ene_ana.md++.lib
@en_files
"""
    for i in range(start,NRUN,1):
            body += f"../aeds_{name}_{i}.tre.gz\n"
    return body

def build_rmsd(settings_loaded, NRUN):
    name = settings_loaded['system']['name']
    topo = settings_loaded['system']['topo_file']
    equilibrate = settings_loaded['simulation']['equilibrate']
    if equilibrate[0] == True: 
        start = NRUN*(equilibrate[1]/100)
        start = round(start)
    else:
        start = 1
    body = f"""@topo ../../../../topo/{topo}
@pbc r cog
@atomsrmsd  1:CA
@atomsfit   1:CA,C,N
@traj
"""
    
    body += f"../aeds_{name}_{start}.trc.gz\n"
    body += f"../aeds_{name}_{NRUN}.trc.gz\n"
    return body

def build_dfmult_file(settings_loaded):
    temp = settings_loaded['simulation']['parameters']['temp']
    body = f"""
@temp {temp}
@stateR eds_vr.dat
@endstates e1.dat e2.dat"""
    return body

def build_output(settings_loaded,fractions,dG,rmsd):
    offsets = settings_loaded['simulation']['parameters']['EIR_list']
    n= len(offsets)
    body = f"""#RUN,OFFSET,FRACTION1,FRACTION2,dG,rmsd\n"""

    for i in range(1,n+1,1):
            body += f"{i},{offsets[i-1]},{fractions[i-1][0]},{fractions[i-1][1]},{dG[i-1]},{rmsd[i-1]}\n"
    return body
