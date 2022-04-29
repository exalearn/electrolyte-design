import sys
from functools import partial, update_wrapper
from queue import Queue
from threading import Event
from typing import List, Set, Tuple
from pathlib import Path
from csv import DictWriter
from datetime import datetime
import argparse
import logging
import json
import bz2

from colmena.redis.queue import make_queue_pairs, ClientQueues
from colmena.task_server import ParslTaskServer
from colmena.thinker import BaseThinker, agent, ResourceCounter
from parsl import Config, HighThroughputExecutor
from parsl.addresses import address_by_hostname
from parsl.launchers import AprunLauncher
from parsl.providers import CobaltProvider
from rdkit import Chem
import proxystore as ps
import numpy as np
import yaml


def parsl_config(name: str) -> Tuple[Config, int]:
    """Make the compute resource configuration

    Args:
        name: Name of the diesred configuration
    Returns:
        - Parsl compute configuration
        - Number of compute slots: Includes execution slots and pre-fetch buffers
    """

    if name == 'local':
        return Config(
            executors=[
                HighThroughputExecutor(max_workers=16, prefetch_capacity=1)
            ]
        ), 64
    elif name == 'theta-debug':
        return Config(
            retries=16,
            executors=[HighThroughputExecutor(
                    address=address_by_hostname(),
                    label="debug",
                    max_workers=64,
                    prefetch_capacity=64,
                    cpu_affinity='block',
                    provider=CobaltProvider(
                        account='redox_adsp',
                        queue='debug-flat-quad',
                        nodes_per_block=8,
                        scheduler_options='#COBALT --attrs enable_ssh=1',
                        walltime='00:60:00',
                        init_blocks=0,
                        max_blocks=1,
                        cmd_timeout=360,
                        launcher=AprunLauncher(overrides='-d 64 --cc depth -j 1'),
                        worker_init='''
module load miniconda-3
conda activate /lus/theta-fs0/projects/CSC249ADCD08/edw/env''',
                    ),
                )]
            ), 64 * 8 * 4
    else:
        raise ValueError(f'Configuration not defined: {name}')


class ScreenEngine(BaseThinker):
    """Screening engine that screens molecules in parallel

    Reads from a list of molecules too large to fit in memory and sends it out gradually in small chunks to be evaluated

    Parameters:
        queues: List of queues
        screen_path: Path to molecules to be screened
        store: Store used to pass inputs from thinker to worker
        output_dir: Path to the output directory
        slot_count: Number of execution slots
        chunk_size: Number of molecules per chunk
    """

    def __init__(self,
                 queues: ClientQueues,
                 store: ps.store.remote.RemoteStore,
                 screen_paths: List[Path],
                 output_dir: Path,
                 slot_count: int,
                 chunk_size: int
                 ):
        # Init the base class
        super().__init__(queues, ResourceCounter(slot_count, ['screen']))
        self.rec.reallocate(None, 'screen', 'all')

        # Store the input and output information
        self.screen_paths = screen_paths
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.store = store

        # Queue to store ready-to-compute chunks
        self.screen_queue = Queue(slot_count * 2)

        # Thinks to know if we are done
        self.all_read = Event()
        self.total_chunks = 0
        self.total_molecules = 0

    @agent(startup=True)
    def read_chunks(self):
        """Read chunks to be screened"""
        
        # Make a function to push a chunk of molecules to the execution queue
        def _proxy_and_push(chunk):
            # Submit the object to the proxy store
            key = f'chunk-{self.total_chunks}'
            chunk_proxy = self.store.proxy(chunk, key=key)
            
            # Increment the count
            self.total_chunks += 1
            
            # Put the proxy and the key in the queue
            self.screen_queue.put((chunk_proxy, key))
        
        # Loop over all files
        chunk = []
        for i, path in enumerate(self.screen_paths):
            self.logger.info(f'Reading from file {i+1}/{len(self.screen_paths)}: {path}')
            
            # Open the file
            if path.suffix == '.bz2':
                fp = bz2.open(path, 'rt')
            else:
                fp = path.open()
            
            # Start the chunks
            chunk = []
            for line in fp:
                try:
                    if path.suffix == '.csv':
                        # The line is comma-separated with the last entry as the string
                        _, smiles = line.strip().rsplit(",", 1)
                    elif path.suffix == '.smi':
                        # The line is just a SMILES string
                        smiles = line.strip()
                    elif path.suffix == '.bz2':
                        # The SMILES string is the first entry
                        smiles, _ = line.split("\t", 1)
                    else:
                        raise ValueError(f'Extension "{path.suffix}" not recognized for "')
                except:
                    raise
                self.total_molecules += 1

                # Add to the chunk and submit if we hit the target size
                chunk.append(smiles)
                if len(chunk) >= self.chunk_size:
                    _proxy_and_push(chunk)
                    chunk = []
            fp.close()

        _proxy_and_push(chunk)

        # Put a None at the end to signal we are done
        self.screen_queue.put(None)

        # Mark that we are done reading
        self.logger.info(f'Finished reading {self.total_molecules} molecules and submitting {self.total_chunks} blocks')
        self.all_read.set()

    @agent(startup=True)
    def submit_task(self):
        """Submit chunks of molecules to be screened"""
        while True:
            # Get the next chunk and, if None, break
            msg = self.screen_queue.get()
            if msg is None:
                break
                
            # Create a proxy for the chunk data
            chunk, key = msg

            # Submit once we have resources available
            self.rec.acquire("screen", 1)
            self.queues.send_inputs(chunk, method='screen_molecules', task_info={'key': key})

    @agent()
    def receive_results(self):
        """Write the screened molecules to disk"""

        # Open the output file
        start_time = np.inf  # Use the time the first compute starts
        num_recorded = 0
        total_passed = 0
        with open(self.output_dir / "screened_molecules.csv", 'w') as fp, open(self.output_dir / 'inference-results.json', 'w') as fq:
            writer = DictWriter(fp, ['inchi', 'smiles', 'molwt', 'dict'])
            writer.writeheader()
            while True:
                # Stop when all have been recorded
                if self.all_read.is_set() and num_recorded >= self.total_chunks:
                    break

                # Wait for a chunk to be received
                result = self.queues.get_result()
                self.rec.release("screen", 1)
                if not result.success:
                    self.logger.error(f'Task failed. Traceback: {result.failure_info.traceback}')
                    raise ValueError('Failed task')
                    
                # Update the start time
                start_time = min(start_time, result.time_compute_started)

                # Write the molecules in the chunk to disk
                #  TODO (wardlt): Make this asynchronous, so we don't block sending new tasks while writing others out
                for row in result.value:
                    writer.writerow(row)
                    total_passed += 1
                num_recorded += 1
                
                # Remove the chunk from the proxystore
                self.store.evict(result.task_info['key'])
                
                # Save the JSON result
                print(result.json(exclude={'inputs', 'value'}), file=fq)

                # Print a status message
                if self.all_read.is_set():
                    self.logger.info(f'Recorded task {num_recorded}/{self.total_chunks}. Molecules so far: {total_passed}')
                else:
                    self.logger.info(f'Recorded task {num_recorded}/???. Molecules so far: {total_passed}')

        # Print the final status
        self.logger.info(f'{total_passed}/{self.total_molecules} ({total_passed / self.total_molecules * 100:.1f}%) passed the screens')
        run_time = datetime.now().timestamp() - start_time
        self.logger.info(f'Runtime {run_time:.2f} s. Evaluation rate: {self.total_molecules / run_time:.3e} mol/s')


def screen_molecules(
    to_screen: List[str],
    max_molecular_weight: float,
    forbidden_smarts: List[str],
    required_smarts: List[str],
    allowed_elements: Set[str],
) -> List[str]:
    """Screen molecules that pass molecular weights and substructure filters

    Args:
        to_screen: List of SMILES strings to string
        max_molecular_weight: Maximum molecular weight (g/mol)
        forbidden_smarts: List of SMARTS that cannot appear in a molecule
        required_smarts: List of SMARTS that must appear in the molecule
        allowed_elements: List of allowed elements
    """
    import json
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from moldesign.utils.conversions import convert_rdkit_to_dict

    # Pre-parse the SMARTS strings
    forbidden_smarts = [Chem.MolFromSmarts(s) for s in forbidden_smarts]
    required_smarts = [Chem.MolFromSmarts(s) for s in required_smarts]

    passed = []
    for smiles in to_screen:
        mol = Chem.MolFromSmiles(smiles)

        # Skip if molecule does not parse
        if mol is None:
            continue

        # Skip if molecular weight is above a threshold
        mol_wt = Descriptors.MolWt(mol)
        if mol_wt > max_molecular_weight:
            continue

        # Skip if it contains a disallowed elements
        if any(atom.GetSymbol() not in allowed_elements for atom in mol.GetAtoms()):
                continue
        
        # Skip if it contains a disallowed group
        try:
            if any(mol.HasSubstructMatch(s) for s in forbidden_smarts):
                continue
        except:
            continue
        
        # Skip if it does not contain all of the allowed groups
        try:
            if not all(mol.HasSubstructMatch(s) for s in required_smarts):
                continue
        except:
            continue
            
        # Parse it into dictionary format
        mol = Chem.AddHs(mol)
        moldict = convert_rdkit_to_dict(mol)
        
        # Prepare it to be JSON serialized
        moldict = dict((k, v.tolist() if isinstance(v, np.ndarray) else v) for k, v in moldict.items())
        
        # Add it to the output dictionary
        passed.append({
            'smiles': smiles,
            'inchi': Chem.MolToInchi(mol),
            'molwt': mol_wt,
            'dict': json.dumps(moldict)
        })

    return passed


if __name__ == '__main__':
    # User inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--redishost", default="127.0.0.1",
                        help="Address at which the redis server can be reached")
    parser.add_argument("--redisport", default="6379",
                        help="Port on which redis is available")
    parser.add_argument('--search-spaces', nargs='+', help='Path(s) to molecules to be screened', required=True)
    parser.add_argument('--name', help='Name for the run')
    parser.add_argument('--max-molecular-weight', default=150, help='Maximum allowed molecular weight. Units: g/mol. Default based on doi: 10.1039/C4EE02158D',
                        type=float)
    parser.add_argument("--molecules-per-chunk", default=50000, type=int, help="Number molecules per screening task")
    parser.add_argument("-s", "--screening-parameters", default=None, help='Path to list of allowed elements and forbidden groups')
    parser.add_argument('--compute', default='local', help='Resource configuration')
    parser.add_argument('--proxy-store', default='file', help='What kind of proxy store to use')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite a previous run')

    # Parse the arguments
    args = parser.parse_args()
    run_params = args.__dict__

    # Check that the search path exists
    search_paths = [Path(x) for x in args.search_spaces]
    assert all(x.is_file() for x in search_paths)

    # Load in the name screening parameters
    screen_params_path = Path(args.screening_parameters)
    with screen_params_path.open() as fp:
        screen_params = yaml.load(fp, yaml.SafeLoader)

    # Make sure all the substructure SMARTS are valid
    for smarts in screen_params['bad_smarts']:
        if Chem.MolFromSmarts(smarts) is None:
            raise ValueError(f'Invalid SMARTS: {smarts}')
            
    # Determine a name for the name
    name = search_paths[0].name[:-4]
    if args.name is not None:
        name = args.name
    
    # Create an output directory with the name of the directory
    out_path = Path().joinpath('runs', f'{name}-molwt={args.max_molecular_weight}-params={screen_params_path.name[:-4]}')
    out_path.mkdir(parents=True, exist_ok=args.overwrite)
    
    # Store the run parameters
    with open(out_path / 'screen-params.yaml', 'w') as fp:
        fp.write(screen_params_path.read_text())
    with open(out_path / 'run-config.json', 'w') as fp:
        json.dump(run_params, fp, indent=2)
    
    # Set up the logging
    handlers = [logging.FileHandler(out_path / 'runtime.log', mode='w'), logging.StreamHandler(sys.stdout)]


    class ParslFilter(logging.Filter):
        """Filter out Parsl debug logs"""

        def filter(self, record):
            return not (record.levelno == logging.DEBUG and '/parsl/' in record.pathname)


    for h in handlers:
        h.addFilter(ParslFilter())

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO, handlers=handlers)

    # Prepare the screening function
    screen_fun = partial(screen_molecules, 
                         max_molecular_weight=args.max_molecular_weight,
                         forbidden_smarts=screen_params['bad_smarts'],
                         required_smarts=screen_params.get('required_smarts', []),
                         allowed_elements=set(screen_params['allowed_elements']))
    update_wrapper(screen_fun, screen_molecules)

    # Make Parsl engine
    config, n_slots = parsl_config(args.compute)

    # Configure the file 
    if args.proxy_store == 'file':
        ps_file_dir = out_path / 'file-store'
        ps_file_dir.mkdir(exist_ok=True)
        store = ps.store.init_store(ps.store.STORES.FILE, name='file', store_dir=str(ps_file_dir))
    elif args.proxy_store == 'redis':
        store = ps.store.init_store(ps.store.STORES.REDIS, name='redis', hostname=args.redishost, port=args.redisport)
    else:
        raise ValueError('ProxyStore config not recognized: {}')

    # Make the task queues and task server
    client_q, server_q = make_queue_pairs(args.redishost, args.redisport, name='jscreen', 
                                          keep_inputs=False, serialization_method='pickle',
                                          proxystore_threshold=1000, proxystore_name=store.name)
    task_server = ParslTaskServer([screen_fun], server_q, config)

    # Make the thinker
    thinker = ScreenEngine(client_q, store, search_paths, out_path, n_slots, args.molecules_per_chunk)

    # Run the program
    try:
        task_server.start()
        thinker.run()
    finally:
        client_q.send_kill_signal()

    task_server.join()
    store.cleanup()
