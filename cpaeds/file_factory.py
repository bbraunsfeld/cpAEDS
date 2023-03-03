import datetime
import os
import re

class Block:
    '''
    TODO: add a translation layer, translating value names to indices in the corresponding blocks.
    Stores an imd block and can be used to change values in the block.
    '''

    def __init__(self, blockname: str, contentRaw: str) -> None:
        self.blockname = blockname
        self.content = list()
        self.isComment = list()

        # Parses the imd block and puts it into the content list, and finds out if it the line was a comment or not.
        for l in contentRaw.split('\n'):

            if re.match("^#", l) or l == "" or self.blockname == "TITLE":
                self.isComment.append(True)
                self.content.append([l])
            else:
                self.isComment.append(False)
                self.content.append(l.split())

    def __repr__(self) -> str:
        return self.getBlock()

    def changeValueByIndex(self, index: int, value):
        '''
        Changes a value in the block to the given value
        index: the index of the value to be changed. Counting starts with 0 at the beginning of each block.
        value: the new value the value at the given position should be changed to

        Retruns None
        '''

        i = 0
        for nOuter, (line, cmnt) in enumerate(zip(self.content, self.isComment)):
            if not cmnt:
                for nInner, oldValue in enumerate(line):
                    if i == index:
                        self.content[nOuter][nInner] = str(value)
                        return None
                    i += 1
        
        raise IndexError(f"Index {index} in block {self.blockname} is out of bounds. Check positions.")
        

    def getBlock(self) -> str:
        # Returns a (more or less) neatly formatted block.
        blockout = self.blockname + "\n"

        for contentOuter in self.content:
            for inner in contentOuter:
                blockout += inner + "\t"
            blockout += "\n"

        blockout = blockout[:-2]
        blockout += "END\n"
        return blockout

class IMDFile:
    ''' 
    Holds the content of an IMD file and allows for manipulation
    '''

    def __init__(self, inFile: str) -> None:

        dictFile = os.path.abspath(os.path.join(os.path.dirname(__file__), f"data/IMDNames.lib"))
        aedsTemplate = os.path.abspath(os.path.join(os.path.dirname(__file__), f"data/AEDS.template"))

        self.blocks = dict() # Holds the blocks with the blockname as key and the content as Block object
        self.valueIndices = dict() # Holds the positions and names of values. This needs to be hardcoded. TODO: change it over to a library file?

        # Read blocks from file
        self.readBlocksFromFile(inFile)

        if not ('AEDS' in self.blocks.keys()):
            # Adds an AEDS block from aeds.template
            self.readBlocksFromFile(aedsTemplate)

        with open(dictFile, "r") as f:
            for l in f:
                if not re.match("^#", l):
                    name, block, index = l.strip().split(",")
                    self.valueIndices[name] = [block, int(index)]

        return None

    def readBlocksFromFile(self, inFile):
        with open(inFile, "r") as f:
            line = f.readline()
            while line:
                # Iterate over all lines, if a new block was found by the regex, the name and content is stored.

                if re.match("^[A-Z]+\n", line):
                    if not re.match("^END\n", line):
                        currentBlock = line.strip()
                        tempContent = str()
                    else:
                        self.blocks[currentBlock] = Block(currentBlock, tempContent) 
                else:
                     if not re.match("^END", line): tempContent += line

                line = f.readline()
            # Need this to add the last block if there is no newline after the last END
            # If there is a newline, it has no consequences.
            self.blocks[currentBlock] = Block(currentBlock, tempContent)

    def __repr__(self):
        return "".join(self.getIMDforAEDS())

    def changeValueByIndex(self, blockname: str, index: int, value):
        self.blocks[blockname].changeValueByIndex(index, value)

    
    def changeValueByName(self, ValueName: str, value) -> None:
        ''' 
        Changes an value by the given name accoring to GROMOS Volume XX
        Currently only works for few named values.
        '''
        self.changeValueByIndex(*self.valueIndices[ValueName], value)

        return None

    def getIMDforAEDS(self) -> list:
        #Returns all block except the block "POSITIONRES" as a list of lines. Entries contain \n.
        out = [block.getBlock() for block in self.blocks.values() if not block.blockname == 'POSITIONRES']
        return out

    def writeIMDforAEDS(self, dest) -> None:
        with open(dest, "w") as f:
            f.writelines(self.getIMDforAEDS())


def build_mk_script_file(settings_loaded,dir_path):
    name = settings_loaded['system']['name']
    md_engine = settings_loaded['system']['path_to_md_engine']
    topo = os.path.relpath(settings_loaded['system']['topo_file'], dir_path)
    #md_dir_name = os.path.basename(os.path.normpath(settings_loaded['system']['md_dir']))
    cnf = os.path.relpath(f"{settings_loaded['system']['md_dir']}/{settings_loaded['system']['cnf_file']}",  dir_path)
    pert =  os.path.relpath(settings_loaded['system']['pert_file'], dir_path)
    version = settings_loaded['system']['version']
    if settings_loaded['system']['lib_type'] == f"cuda":
        lib = f'mk_script_cuda_8_slurm.lib'
    elif settings_loaded['system']['lib_type'] == f"cuda_local":
        lib=f'mk_script_cuda_8.lib'
    body = f"""@sys            aeds_{name}
@bin            {md_engine}
@dir            {dir_path}
@files
  topo          {topo}
  input         aeds.imd
  coord         {cnf}
  pttopo        {pert}
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

                    
def build_imd_file(settings_loaded,EIR,rs):

    ref_imd_path = f"{settings_loaded['system']['md_dir']}/{settings_loaded['system']['ref_imd']}"

    NUMSTATES = settings_loaded['simulation']['NSTATES']
    NSTLIM = settings_loaded['simulation']['parameters']['NSTLIM']
    NTPR = settings_loaded['simulation']['parameters']['NTPR']
    NTWX = settings_loaded['simulation']['parameters']['NTWX']
    NTWE = settings_loaded['simulation']['parameters']['NTWE']
    dt = settings_loaded['simulation']['parameters']['dt']
    EMIN = settings_loaded['simulation']['parameters']['EMIN']
    EMAX = settings_loaded['simulation']['parameters']['EMAX']

    imd = IMDFile(ref_imd_path)
    imd.changeValueByName('NUMSTATES', NUMSTATES)
    imd.changeValueByName('NSTLIM', NSTLIM)
    imd.changeValueByName('NTPR', NTPR)
    imd.changeValueByName('NTWX', NTWX)
    imd.changeValueByName('NTWE', NTWE)
    imd.changeValueByName('dt', dt)
    imd.changeValueByName('EMIN', EMIN)
    imd.changeValueByName('EMAX', EMAX)
    imd.changeValueByName('EIR', '\t'.join(str(i) for i in EIR)) #loops over elements in EIR (list of offsets at the same level) 
    imd.changeValueByName('T', 0)

    if settings_loaded['system']['lib_type'] == f"cuda":
        ALPHLJ='0'  
        ALPHCRF='0'
    elif settings_loaded['system']['lib_type'] == f"cuda_local":
        ALPHLJ=''  
        ALPHCRF=''

    imd.changeValueByName('ALPHLJ', ALPHLJ)
    imd.changeValueByName('ALPHCRF', ALPHCRF)

    return "".join(imd.getIMDforAEDS())
    

def build_ene_ana(settings_loaded,NRUN):
    dir_path = os.getcwd()
    name = settings_loaded['system']['name']
    nstates = settings_loaded['simulation']['NSTATES']
    topo = os.path.relpath(settings_loaded['system']['topo_file'], dir_path)
    equilibrate= settings_loaded['simulation']['equilibrate']
    if equilibrate[0] == True:
        start = NRUN*(equilibrate[1]/100)
        start = round(start)
    else:
        start = 1
    NRUN = NRUN + 1
    header = f"""@prop eds_vr eds_emin eds_emax eds_vmix eds_globmin eds_globminfluc """
    for i in range(nstates):
        header += f"e{i+1} e{i+1}s "
    body = header + f"""\n@topo {topo}
@library ene_ana.md++.lib
@en_files
"""
    for i in range(start,NRUN,1):
            body += f"../aeds_{name}_{i}.tre.gz\n"
    return body

def build_rmsd(settings_loaded, NRUN):
    dir_path = os.getcwd()
    name = settings_loaded['system']['name']
    topo = os.path.relpath(settings_loaded['system']['topo_file'], dir_path)
    equilibrate = settings_loaded['simulation']['equilibrate']
    if equilibrate[0] == True: 
        start = NRUN*(equilibrate[1]/100)
        start = round(start)
    else:
        start = 1
    body = f"""@topo {topo}
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
    endstates = f"""@endstates """
    for i in range(settings_loaded['simulation']['NSTATES']):
        endstates += f"e{i+1}.dat "
    body = f"""
@temp {temp}
@stateR eds_vr.dat
{endstates}"""
    return body

def build_output(settings_loaded,fractions,dF,rmsd):
    #temporary fix to select first in offset list. Needs changes for multi state cpAEDS
    offsets = settings_loaded['simulation']['parameters']['EIR_list']
    n= len(offsets[0])
    header = f"""#RUN,"""
    offset_header = f""""""
    fraction_header = f""""""
    dF_header = f""""""
    for i in range(settings_loaded['simulation']['NSTATES']):
        offset_header += f"OFFSET{i+1},"
        fraction_header += f"FRACTION{i+1},"
        dF_header += f"dF{i+1},"
    header = header + offset_header + fraction_header + dF_header + f"rmsd\n"
    body = f""""""
    for i in range(1,n+1,1):
            body += f"{i},0,"
            for j in offsets:
                body += f"{j[i-1]},"
            for j in fractions[i-1]:
                body += f"{j},"
            for j in dF[i-1]:
                body += f"{j},"
            body += f"{rmsd[i-1]}\n"
    file = header + body        
    return file

