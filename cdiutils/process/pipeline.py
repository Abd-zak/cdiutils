from typing import Callable, Dict
from string import Template
import glob
import os
import shutil
import subprocess
import time
import traceback

import paramiko
import ruamel.yaml
import yaml

from cdiutils.plot.formatting import update_plot_params
from cdiutils.utils import pretty_print
from .processor import BcdiProcessor
from .phaser import PhasingResultAnalyser
from .parameters import check_parameters, convert_np_arrays

from cdiutils.process.facet_analysis import FacetAnalysisProcessor

try:
    from bcdi.preprocessing.preprocessing_runner import (
        run as run_preprocessing
    )
    from bcdi.postprocessing.postprocessing_runner import (
        run as run_postprocessing
    )
    from bcdi.utils.parser import ConfigParser
    IS_BCDI_AVAILABLE = True
except ModuleNotFoundError:
    print("The bcdi package is not installed. bcdi backend won't be available")
    IS_BCDI_AVAILABLE = False  # is_bcdi_available

BCDI_ERROR_TEXT = (
    "Cannot use 'bcdi' backend if bcdi package is not"
    "installed."
)


def make_scan_parameter_file(
        output_parameter_file_path: str,
        parameter_file_template_path: str,
        updated_parameters: dict
) -> None:
    """
    Create a scan parameter file given a template and the parameters
    to update.
    """

    with open(parameter_file_template_path, "r", encoding="utf8") as file:
        source = Template(file.read())

    scan_parameter_file = source.substitute(updated_parameters)

    with open(output_parameter_file_path, "w", encoding="utf8") as file:
        file.write(scan_parameter_file)


def update_parameter_file(file_path: str, updated_parameters: dict) -> None:
    """
    Update a parameter file with the provided dictionary that contains
    the parameters (keys, values) to uptade.
    """
    convert_np_arrays(updated_parameters)
    with open(file_path, "r", encoding="utf8") as file:
        config, ind, bsi = ruamel.yaml.util.load_yaml_guess_indent(file)

    for key in config.keys():
        for updated_key, updated_value in updated_parameters.items():
            if updated_key in config[key]:
                config[key][updated_key] = updated_value
            elif updated_key == key:
                config[key] = updated_value
            else:
                for sub_key in config[key].keys():
                    if (
                            isinstance(config[key][sub_key], dict)
                            and updated_key in config[key][sub_key]
                    ):
                        config[key][sub_key][updated_key] = updated_value

    yaml_file = ruamel.yaml.YAML()
    yaml_file.indent(mapping=ind, sequence=ind, offset=bsi)
    with open(file_path, "w", encoding="utf8") as file:
        yaml_file.dump(config, file)


if IS_BCDI_AVAILABLE:
    class BcdiPipelineParser(ConfigParser):
        def __init__(self, file_path: str) -> None:
            super().__init__(file_path)

        def load_arguments(self) -> Dict:
            raw_args = yaml.load(self.raw_config, Loader=yaml.SafeLoader)

            raw_args["preprocessing"].update(raw_args["general"])
            raw_args["postprocessing"].update(raw_args["general"])
            raw_args["pynx"].update(
                {"detector_distance":
                    raw_args["general"]["detector_distance"]}
            )

            self.arguments = {
                "preprocessing": self._check_args(raw_args["preprocessing"]),
                "pynx": raw_args["pynx"],
                "postprocessing": self._check_args(raw_args["postprocessing"]),
            }
            try:
                self.arguments["cdiutils"] = raw_args["cdiutils"]
            except KeyError:
                print("No cdiutils arguments given")
            return self.arguments

        def load_bcdi_parameters(
                self,
                procedure: str = "preprocessing"
        ) -> Dict:
            raw_args = yaml.load(
                self.raw_config,
                Loader=yaml.SafeLoader
            )[procedure]
            raw_args.update(raw_args["general"])
            return self._check_args(raw_args)


def process(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as exc:
            print(
                "\n[ERROR] An error occured in the "
                f"'{func.__name__}' method... here is the traceback:\n"
            )
            traceback.print_exception(exc)
            raise
    return wrapper


class BcdiPipeline:
    """
    A class to handle the bcdi workflow, from pre-processing to
    post-processing (bcdi package), including phase retrieval
    (pynx package).

    :param parameter_file_path: the path (str) of the scan parameter
    file that holds all the information related to the entire process.
    """
    def __init__(
            self,
            parameter_file_path: str = None,
            parameters: dict = None,
            backend: str = "cdiutils"
    ):

        self.parameter_file_path = parameter_file_path
        self.params = parameters

        if parameters is None:
            if parameter_file_path is None:
                raise ValueError(
                    "parameter_file_path or parameters must be provided"
                )
            self.params = self.load_parameters(backend)
        else:
            check_parameters(parameters)

        self.backend = backend

        if backend == "cdiutils":
            self.dump_dir = (
                self.params["cdiutils"]["metadata"]["dump_dir"]
            )
            self.scan = self.params["cdiutils"]["metadata"]["scan"]
            self.sample_name = (
                self.params["cdiutils"]["metadata"]["sample_name"]
            )
            self.pynx_phasing_dir = self.dump_dir + "/pynx_phasing/"
        elif backend == "bcdi":
            if not IS_BCDI_AVAILABLE:
                raise ModuleNotFoundError(BCDI_ERROR_TEXT)
            self.dump_dir = self.params["preprocessing"][
                "save_dir"][0]
            self.scan = self.params['preprocessing']['scans'][0]
        else:
            raise ValueError(
                f"[ERROR] Unknown backend value ({backend}), it must be either"
                " 'cdiutils' or 'bcdi'"
            )

        # the bcdi_processor attribute will be used only if backend
        # is cdiutils
        self.bcdi_processor: BcdiProcessor = None
        self.result_analyser: PhasingResultAnalyser = None

        # update the plot parameters
        update_plot_params(
            usetex=self.params["cdiutils"]["usetex"],
            use_siunitx=self.params["cdiutils"]["usetex"],
            **{
                "axes.labelsize": 7,
                "xtick.labelsize": 6,
                "ytick.labelsize": 6,
                "figure.titlesize": 8,
            }
        )

    def load_parameters(
            self,
            backend: str = None,
            file_path: str = None
    ) -> dict:
        """
        Load the parameters from the configuration files.
        """
        if backend is None:
            backend = self.backend
        if file_path is None:
            file_path = self.parameter_file_path

        if backend == "bcdi":
            return BcdiPipelineParser(
                file_path
            ).load_arguments()

        if backend == "cdiutils":
            with open(file_path, "r", encoding="utf8") as file:
                parameters = yaml.load(
                    file,
                    Loader=yaml.FullLoader
                )
            check_parameters(parameters)
            return parameters

        raise ValueError(
            f"[ERROR] Unknwon backend value ({backend}), it must be either"
            " 'cdiutils' or 'bcdi'"
        )

    @process
    def preprocess(self, backend: str = None) -> None:

        if backend is None:
            backend = self.backend

        if backend == "bcdi":
            if not IS_BCDI_AVAILABLE:
                raise ModuleNotFoundError(BCDI_ERROR_TEXT)
            os.makedirs(
                self.params["preprocessing"]["save_dir"][0],
                exist_ok=True
            )
            pretty_print(
                "[INFO] Proceeding to bcdi preprocessing using the bcdi "
                f"backend ({self.sample_name}, S{self.scan})"
            )
            run_preprocessing(prm=self.params["preprocessing"])
            pynx_input_template = "S*_pynx_norm_*.npz"
            pynx_mask_template = "S*_maskpynx_norm_*.npz"
            self.save_parameter_file()

        elif backend == "cdiutils":
            pretty_print(
                "[INFO] Proceeding to preprocessing"
                f" ({self.sample_name}, S{self.scan})"
            )
            dump_dir = self.params["cdiutils"]["metadata"]["dump_dir"]
            if os.path.isdir(dump_dir):
                print(
                    "\n[INFO] Dump directory already exists, results will be "
                    f"saved in:\n{dump_dir}."
                )
            else:
                print(
                    f"[INFO] Creating the dump directory at: {dump_dir}")
                os.makedirs(
                    dump_dir,
                    exist_ok=True
                )
            os.makedirs(self.pynx_phasing_dir, exist_ok=True)
            self.bcdi_processor = BcdiProcessor(
                parameters=self.params["cdiutils"]
            )
            self.bcdi_processor.preprocess_data()
            self.bcdi_processor.save_preprocessed_data()
            pynx_input_template = (
                f"{self.pynx_phasing_dir}/S*_pynx_input_data.npz"
            )
            pynx_mask_template = (
                f"{self.pynx_phasing_dir}/S*_pynx_input_mask.npz"
            )

        else:
            raise ValueError(
                f"[ERROR] Unknown backend value ({backend}), it must be either"
                " 'cdiutils' or 'bcdi'"
            )

        try:
            data_path = glob.glob(pynx_input_template)[0]
            mask_path = glob.glob(pynx_mask_template)[0]
        except IndexError as exc:
            raise FileNotFoundError(
                "[ERROR] file missing, something went"
                " wrong during preprocessing"
            ) from exc

        # update the parameters
        if self.parameter_file_path is not None:
            pretty_print("[INFO] Updating scan parameter file")
            update_parameter_file(
                self.parameter_file_path,
                {
                    "data": data_path,
                    "mask": mask_path,
                    "cdiutils": self.bcdi_processor.params
                }
            )

        self.params["cdiutils"].update(self.bcdi_processor.params)
        self.params["pynx"].update({"data": data_path})
        self.params["pynx"].update({"mask": mask_path})
        self.save_parameter_file()
        if self.params["cdiutils"]["show"] and backend == "cdiutils":
            self.bcdi_processor.show_figures()

    @process
    def phase_retrieval(
            self,
            machine: str = None,  #"slurm-nice-devel",
            user: str = os.environ["USER"],
            number_of_nodes: int = 2,
            key_file_path: str = os.environ["HOME"] + "/.ssh/id_rsa",
            pynx_slurm_file_template: str = None,
            clear_former_results: bool = False
    ) -> None:
        """
        Run the phase retrieval using pynx through ssh connection to a
        gpu machine.
        """

        pretty_print(
            "[INFO] Proceeding to PyNX phase retrieval "
            f"({self.sample_name}, S{self.scan})"
        )

        if clear_former_results:
            print("[INFO] Removing former results.\n")
            files = glob.glob(self.pynx_phasing_dir + "/*Run*.cxi")
            files += glob.glob(self.pynx_phasing_dir + "/*Run*.svg")
            for f in files:
                os.remove(f)
            self.phasing_results = []

        pynx_input_file_path = (
            self.pynx_phasing_dir + "/pynx-cdi-inputs.txt"
        )

        # Make the pynx input file
        with open(pynx_input_file_path, "w", encoding="utf8") as file:
            for key, value in self.params["pynx"].items():
                file.write(f"{key} = {value}\n")

        if machine is None:
            print(
                "[INFO] No machine provided, assuming PyNX is installed on "
                "the current machine.\n"
            )
            if os.uname()[1].lower().startswith(("p9", "scisoft16")):
                with subprocess.Popen(
                        # "source /sware/exp/pynx/activate_pynx.sh;"
                        f"cd {self.pynx_phasing_dir};"
                        # "mpiexec -n 4 /sware/exp/pynx/devel.p9/bin/"
                        "pynx-cdi-id01 pynx-cdi-inputs.txt",
                        shell=True,
                        executable="/bin/bash",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                ) as proc:
                    stdout, stderr = proc.communicate()
                    print(
                        "[STDOUT FROM SUBPROCESS RUNNING PYNX]\n",
                        stdout.decode("utf-8")
                    )
                    if proc.returncode:
                        print(
                            "[STDERR FROM SUBPROCESS RUNNING PYNX]\n",
                            stderr.decode("utf-8")
                        )
        else:
            # ssh to the machine and run phase retrieval
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                hostname=machine,
                username=user,
                pkey=paramiko.RSAKey.from_private_key_file(key_file_path)
            )

            print(f"[INFO] Connected to {machine}")
            if machine == "slurm-nice-devel":

                # Make the pynx slurm file
                if pynx_slurm_file_template is None:
                    pynx_slurm_file_template = (
                        f"{os.path.dirname(__file__)}/"
                        "pynx-id01cdi_template.slurm"
                    )
                    print(
                        "Pynx slurm file template not provided, will take "
                        f"the default: {pynx_slurm_file_template}")

                with open(
                        pynx_slurm_file_template, "r", encoding="utf8"
                ) as file:
                    source = Template(file.read())
                    pynx_slurm_text = source.substitute(
                        {
                            "number_of_nodes": number_of_nodes,
                            "data_path": self.pynx_phasing_dir,
                            "SLURM_JOBID": "$SLURM_JOBID",
                            "SLURM_NTASKS": "$SLURM_NTASKS"
                        }
                    )
                with open(
                        self.pynx_phasing_dir + "/pynx-id01cdi.slurm",
                        "w",
                        encoding="utf8"
                ) as file:
                    file.write(pynx_slurm_text)

                # submit job using sbatch slurm command
                _, stdout, _ = client.exec_command(
                    f"cd {self.pynx_phasing_dir};"
                    "sbatch pynx-id01cdi.slurm"
                )
                job_submitted = False
                time.sleep(0.5)

                # read the standard output, decode it and print it
                output = stdout.read().decode("utf-8")

                # get the job id and remove '\n' and space characters
                while not job_submitted:
                    try:
                        job_id = output.split(" ")[3].strip()
                        job_submitted = True
                        print(output)
                    except IndexError:
                        print("Job still not submitted...")
                        time.sleep(3)
                        print(output)
                    except KeyboardInterrupt as err:
                        print("User terminated job with KeyboardInterrupt.")
                        client.close()
                        raise err

                # while loop to check if job has terminated
                process_status = "PENDING"
                while process_status != "COMPLETED":
                    _, stdout, _ = client.exec_command(
                        f"sacct -j {job_id} -o state | head -n 3 | tail -n 1"
                    )

                    # python process needs to sleep here, otherwise it gets in
                    # trouble with standard output management. Anyway, we need
                    # to sleep in the while loop in order to wait for the
                    # remote  process to terminate.
                    time.sleep(2)
                    process_status = stdout.read().decode("utf-8").strip()
                    print(f"[INFO] process status: {process_status}")

                    if process_status == "RUNNING":
                        _, stdout, _ = client.exec_command(
                            f"cd {self.pynx_phasing_dir};"
                            f"cat pynx-id01cdi.slurm-{job_id}.out "
                            "| grep 'CDI Run:'"
                        )
                        time.sleep(1)
                        print(stdout.read().decode("utf-8"))

                    elif process_status == "CANCELLED+":
                        raise RuntimeError("[INFO] Job has been cancelled")
                    elif process_status == "FAILED":
                        raise RuntimeError(
                            "[ERROR] Job has failed. Check out logs at: \n",
                            f"{self.pynx_phasing_dir}/"
                            f"pynx-id01cdi.slurm-{job_id}.out"
                        )

                if process_status == "COMPLETED":
                    print(f"[INFO] Job {job_id} is completed.")

            else:
                _, stdout, stderr = client.exec_command(
                    "source /sware/exp/pynx/activate_pynx.sh 2022.1;"
                    f"cd {self.pynx_phasing_dir};"
                    "pynx-id01cdi.py pynx-cdi-inputs.txt "
                    f"2>&1 | tee phase_retrieval_{machine}.log"
                )
                if stdout.channel.recv_exit_status():
                    raise RuntimeError(
                        f"Error pulling the remote runtime {stderr.readline()}"
                    )
                for line in iter(lambda: stdout.readline(1024), ""):
                    print(line, end="")
            client.close()

    def analyze_phasing_results(self, *args, **kwargs) -> None:
        """
        Deprecated function, shoud use analyse_phasing_results instead.
        """
        from warnings import warn
        warn(
            "analyze_phasing_results is deprecated; use "
            "analyse_phasing_results instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.analyse_phasing_results(*args, **kwargs)

    def analyse_phasing_results(
            self,
            sorting_criterion: str = "mean_to_max",
            plot_phasing_results: bool = True,
            plot_phase: bool = False,
    ) -> None:
        """
        Wrapper for analyse_phasing_results method of
        PhasingResultAnalyser class.

        Analyse the phase retrieval results by sorting them according to
        the sorting_criteion, which must be selected in among:
        * mean_to_max the difference between the mean of the
            Gaussian fitting of the amplitude histogram and the maximum
            value of the amplitude. We consider the closest to the max
            the mean is, the most homogeneous is the amplitude of the
            reconstruction, hence the best.
        * the sharpness the sum of the amplitude within support to
            the power of 4. For reconstruction with similar support,
            lowest values means graeter amplitude homogeneity.
        * std the standard deviation of the amplitude.
        * llk the log-likelihood of the reconstruction.
        * llkf the free log-likelihood of the reconstruction.

        Args:
            sorting_criterion (str, optional): the criterion to sort the
                results with. Defaults to "mean_to_max".
            plot_phasing_results (bool, optional): whether to plot the
                phasing results. Defaults to True.
            plot_phase (bool, optional): whether the phase must be
                plotted. If True, will the phase is plotted with 
                amplitude as opacity. If False, amplitude is plotted
                instead. Defaults to False.

        Raises:
            ValueError: if sorting_criterion is unknown.
        """
        self.result_analyser = PhasingResultAnalyser(
            result_dir_path=self.pynx_phasing_dir
        )

        self.result_analyser.analyse_phasing_results(
            sorting_criterion,
            plot_phasing_results,
            plot_phase
        )

    def select_best_candidates(
            self,
            nb_of_best_sorted_runs: int = None,
            best_runs: list = None
    ) -> None:
        """
        A function wrapper for
        PhasingResultAnalyser.select_best_candidates 

        Args:
            nb_of_best_sorted_runs (int, optional): _description_. Defaults to None.
            best_runs (list, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        if not self.result_analyser:
            raise ValueError(
                "self.result_analyser not initialised yet. Run"
                " self.analyse_pahsing_results() first."
            )
        self.result_analyser.select_best_candidates(
            nb_of_best_sorted_runs,
            best_runs
        )

    @process
    def mode_decomposition(
            self,
            pynx_analysis_script: str = (
                "/cvmfs/hpc.esrf.fr/software/packages/"
                "ubuntu20.04/x86_64/pynx/2024.1/bin/pynx-cdi-analysis"
            ),
            run_command: str = None,
            machine: str = None,
            user: str = None,
            key_file_path: str = None
    ) -> None:
        """
        Run the mode decomposition using PyNX pynx-cdi-analysis.py
        script as a subprocess.

        Args:
            pynx_analysis_script (str, optional): Version of PyNX to
                use. Defaults to "2024.1".
            machine (str, optional): Remote machine to run the mode
                decomposition on. Defaults to None.
            user (str, optional): User for the remote machine. Defaults
                to None.
            key_file_path (str, optional): Path to the key file for SSH
                authentication. Defaults to None.
        """
        if run_command is None:
            run_command = (
                f"cd {self.pynx_phasing_dir};"
                f"{pynx_analysis_script} candidate_*.cxi --modes 1 "
                "--modes_output mode.h5 2>&1 | tee mode_decomposition.log"
            )

        if machine:
            print(f"[INFO] Remote connection to machine '{machine}'requested.")
            if user is None:
                user = os.environ["USER"]
                print(f"user not provided, will use '{user}'.")
            if key_file_path is None:
                key_file_path = os.environ["HOME"] + "/.ssh/id_rsa"
                print(
                    f"key_file_path not provided, will use '{key_file_path}'."
                )

            pretty_print(
                f"[INFO] Running mode decomposition on machine '{machine}'"
                f"({self.sample_name}, S{self.scan})"
            )
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                hostname=machine,
                username=user,
                pkey=paramiko.RSAKey.from_private_key_file(key_file_path)
            )

            _, stdout, stderr = client.exec_command(run_command)
            # read the standard output, decode it and print it
            formatted_stdout = stdout.read().decode("utf-8")
            formatted_stderr = stderr.read().decode("utf-8")
            print("[STDOUT FROM SSH PROCESS]\n")
            print(formatted_stdout)
            print("[STDERR FROM SSH PROCESS]\n")
            print(formatted_stderr)

            if stdout.channel.recv_exit_status():
                raise RuntimeError(
                    f"Error pulling the remote runtime {stderr.readline()}")
            client.close()

        # if no machine provided, run the mode decomposition as a subprocess
        else:
            with subprocess.Popen(
                    run_command,
                    shell=True,
                    executable="/usr/bin/bash",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
            ) as proc:
                stdout, stderr = proc.communicate()
                print("[STDOUT FROM SUBPROCESS]\n", stdout.decode("utf-8"))
                if proc.returncode:
                    print(
                        "[STDERR FROM SUBPROCESS]\n",
                        stderr.decode("utf-8")
                    )

            if self.parameter_file_path is not None:
                if self.backend == "bcdi":
                    update_parameter_file(
                        self.parameter_file_path,
                        {"reconstruction_files":
                            f"{self.pynx_phasing_dir}mode.h5"}
                    )
                else:
                    update_parameter_file(
                        self.parameter_file_path,
                        {"reconstruction_file":
                            f"{self.pynx_phasing_dir}mode.h5"}
                    )
                self.params = self.load_parameters()

    @process
    def postprocess(self, backend: str = None) -> None:

        if backend is None:
            backend = self.backend

        if backend == "bcdi":
            if not IS_BCDI_AVAILABLE:
                raise ModuleNotFoundError(BCDI_ERROR_TEXT)
            pretty_print(
                "[INFO] Running post-processing from bcdi_strain.py "
                f"({self.sample_name}, S{self.scan})"
            )

            run_postprocessing(prm=self.params["postprocessing"])
            self.save_parameter_file()

        elif backend == "cdiutils":

            pretty_print(
                "[INFO] Running post-processing"
                f"({self.sample_name}, S{self.scan})"
            )

            if self.bcdi_processor is None:
                print("BCDI processor is not instantiated yet.")
                if any(
                        p not in self.params["cdiutils"].keys()
                        or self.params["cdiutils"][p] is None
                        for p in (
                            "q_lab_reference", "q_lab_max",
                            "q_lab_com", "det_reference_voxel",
                            "preprocessing_output_shape"
                        )
                ):
                    file_path = (
                            f"{self.dump_dir}"
                            f"S{self.scan}_parameter_file.yml"
                    )
                    print(f"Loading parameters from:\n{file_path}")
                    preprocessing_params = self.load_parameters(
                        file_path=file_path)["cdiutils"]
                    self.params["cdiutils"].update(
                        {
                            "preprocessing_output_shape": preprocessing_params[
                                "preprocessing_output_shape"
                            ],
                            "det_reference_voxel": preprocessing_params[
                                "det_reference_voxel"
                            ],
                            "q_lab_reference": preprocessing_params[
                                "q_lab_reference"
                            ],
                            "q_lab_max": preprocessing_params["q_lab_max"],
                            "q_lab_com": preprocessing_params["q_lab_com"]
                        }
                    )
                self.bcdi_processor = BcdiProcessor(
                    parameters=self.params["cdiutils"]
                )

            self.bcdi_processor.orthogonalize()
            self.bcdi_processor.postprocess()
            self.bcdi_processor.save_postprocessed_data()
            self.save_parameter_file()
            if self.params["cdiutils"]["show"]:
                self.bcdi_processor.show_figures()

        else:
            raise ValueError(
                f"[ERROR] Unknown backend value ({backend}), it must be either"
                " 'cdiutils' or 'bcdi'"
            )

    def save_parameter_file(self) -> None:
        """
        Save the parameter file used during the analysis.
        """

        output_file_path = (
            f"{self.dump_dir}/S{self.scan}_parameter_file.yml"
            # f"{os.path.basename(self.parameter_file_path)}"
        )

        if self.parameter_file_path is not None:
            try:
                shutil.copy(
                    self.parameter_file_path,
                    output_file_path
                )
            except shutil.SameFileError:
                print(
                    "\nScan parameter file saved at:\n"
                    f"{output_file_path}"
                )

        else:
            convert_np_arrays(self.params)
            with open(output_file_path, "w", encoding="utf8") as file:
                yaml.dump(self.params, file)

            print(
                "\nScan parameter file saved at:\n"
                f"{output_file_path}"
            )

    def facet_analysis(self) -> None:
        facet_anlysis_processor = FacetAnalysisProcessor(
            parameters=self.params
        )
        facet_anlysis_processor.facet_analysis()

