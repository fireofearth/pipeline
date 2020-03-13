#================================================================================
#config file content
#--------------------------------------------------------------------------------
config_file_content = '''
__PIPELINE_INFO__: 
    date_last_updated: 
    name: 'Pipeline'
    data_type: 
    author: 'cchen'
    Kronos_version: '2.3.0'
    host_cluster: 
    version: '0.00.1'
    input_type: 
    date_created: '2019-05-07'
    output_type: 
__GENERAL__: 
    docker: '__REQUIRED__'
__SHARED__: 
    anno_loc: 
    patch_loc: 
    img_loc: 
__SAMPLES__: 
    # sample_id:
    #    param1: value1
    #    param2: value2
TASK_CREATE_GROUPS: 
    run: 
        interval_file: 
        requirements: 
            docker: 
        boilerplate: 
        use_cluster: False
        num_cpus: 1
        parallel_run: False
        merge: True
        add_breakpoint: False
        memory: '128G'
        forced_dependencies: ['TASK_SORT_PATCHES_BY_LABEL']
        env_vars: 
        parallel_params: []
    reserved: 
        # do not change this section.
        seed_version: '0.00.1'
        component_version: '0.00.1'
        component_name: 'kronos_component_singularity'
    component: 
        input_files: 
            slurm_max_time: '10:00:00'
            slurm_error: '/home/cochen/cchen/ml/slurm/create_groups.%j.err'
            slurm_machine: 'dlhost02'
            slurm_job_name: 'create_groups_slurm'
            slurm_num_cpu: 2
            slurm_num_gpu: 2
            slurm_output: '/home/cochen/cchen/ml/slurm/create_groups.%j.out'
            singularity_image: '/home/cochen/cchen/ml/docker_create_groups/create_groups_latest.sif'
            config_file_location: '/projects/ovcare/classification/cchen/ml/pipeline/groups_endometrial_config.json'
            slurm_queue: 'dgxV100'
        parameters: 
        output_files: 
TASK_CREATE_TRAINING_FILES: 
    run: 
        interval_file: 
        requirements: 
            docker: 
        boilerplate: 
        use_cluster: False
        num_cpus: 1
        parallel_run: False
        merge: True
        add_breakpoint: False
        memory: '128G'
        forced_dependencies: ['TASK_CREATE_GROUPS']
        env_vars: 
        parallel_params: []
    reserved: 
        # do not change this section.
        seed_version: '0.00.1'
        component_version: '0.00.1'
        component_name: 'kronos_component_singularity'
    component: 
        input_files: 
            slurm_error: '/home/cochen/cchen/ml/slurm/create_training_files.%j.err'
            chunk_file_location: '/projects/ovcare/classification/cchen/ml/data/endometrial/chunks.txt'
            linker_location: '/projects/ovcare/classification/cchen'
            output_location: '/projects/ovcare/classification/cchen/ml/ml/data/endometrial/training_chunks.txt'
            seed: 1
            singularity_image: '/home/cochen/cchen/ml/docker_create_training_files/create_training_files_latest.sif'
            validation_chunks: '2'
            number_of_patches: '9999999999_9999999999_9999999999'
            slurm_max_time: '10:00:00'
            slurm_machine: 'dlhost02'
            training_chunks: '0_1'
            slurm_job_name: 'create_training_files_slurm'
            slurm_num_cpu: 2
            slurm_num_gpu: 2
            slurm_output: '/home/cochen/cchen/ml/slurm/create_training_files.%j.out'
            slurm_queue: 'dgxV100'
            testing_chunks: '2'
        parameters: 
        output_files: 
TASK_DIVIDE_WORK: 
    run: 
        interval_file: 
        requirements: 
            docker: 
        boilerplate: 
        use_cluster: False
        num_cpus: 1
        parallel_run: False
        merge: True
        add_breakpoint: False
        memory: '128G'
        forced_dependencies: []
        env_vars: 
        parallel_params: []
    reserved: 
        # do not change this section.
        seed_version: '0.00.1'
        component_version: '0.00.1'
        component_name: 'kronos_component_singularity'
    component: 
        input_files: 
            work_location: '/projects/ovcare/classification/cchen/ml/outputs/endometrial/work'
            slurm_max_time: '10:00:00'
            slurm_error: '/home/cochen/cchen/ml/slurm/divide_work.%j.err'
            slurm_machine: 'dlhost02'
            linker_location: '/projects/ovcare/classification/cchen'
            slurm_job_name: 'divide_work_slurm'
            slurm_num_cpu: 2
            'N': 3
            slurm_num_gpu: 2
            slurm_output: '/home/cochen/cchen/ml/slurm/divide_work.%j.out'
            singularity_image: '/home/cochen/cchen/ml/docker_divide_work/divide_work_latest.sif'
            slurm_queue: 'dgxV100'
            data_location: '/projects/ovcare/classification/cchen/ml/data/endometrial/patches'
        parameters: 
        output_files: 
TASK_DOWNSAMPLE_DATA: 
    run: 
        interval_file: 
        requirements: 
            docker: 
        boilerplate: 
        use_cluster: False
        num_cpus: 1
        parallel_run: False
        merge: True
        add_breakpoint: False
        memory: '128G'
        forced_dependencies: ['TASK_DIVIDE_WORK']
        env_vars: 
        parallel_params: []
    reserved: 
        # do not change this section.
        seed_version: '0.00.1'
        component_version: '0.00.1'
        component_name: 'kronos_component_singularity'
    component: 
        input_files: 
            slurm_max_time: '10:00:00'
            slurm_error: '/home/cochen/cchen/ml/slurm/downsample_data.%j.err'
            slurm_machine: 'dlhost02'
            slurm_job_name: 'downsample_data_slurm'
            slurm_num_cpu: 8
            slurm_num_gpu: 2
            slurm_output: '/home/cochen/cchen/ml/slurm/downsample_data.%j.out'
            singularity_image: '/home/cochen/cchen/ml/docker_downsample_data/downsample_data_latest.sif'
            config_file_location: '/projects/ovcare/classification/cchen/ml/pipeline/downsample_endometrial_config.json'
            slurm_queue: 'dgxV100'
        parameters: 
        output_files: 
TASK_SORT_PATCHES_BY_LABEL: 
    run: 
        interval_file: 
        requirements: 
            docker: 
        boilerplate: 
        use_cluster: False
        num_cpus: 1
        parallel_run: False
        merge: True
        add_breakpoint: False
        memory: '128G'
        forced_dependencies: ['TASK_DOWNSAMPLE_DATA']
        env_vars: 
        parallel_params: []
    reserved: 
        # do not change this section.
        seed_version: '0.00.1'
        component_version: '0.00.1'
        component_name: 'kronos_component_singularity'
    component: 
        input_files: 
            patch_location: '/projects/ovcare/classification/cchen/ml/data/endometrial/patches_downsampled'
            slurm_max_time: '10:00:00'
            slurm_error: '/home/cochen/cchen/ml/slurm/sort_patches_by_label.%j.err'
            slurm_machine: 'dlhost02'
            slurm_job_name: 'sort_patches_by_label_slurm'
            sorted_location: '/projects/ovcare/classification/cchen/ml/data/endometrial/patches_downsampled_sorted'
            slurm_num_cpu: 2
            slurm_num_gpu: 2
            slurm_output: '/home/cochen/cchen/ml/slurm/sort_patches_by_label.%j.out'
            labels_file_location: '/projects/ovcare/classification/cchen/ml/data/endometrial/labels_file.txt'
            singularity_image: '/home/cochen/cchen/ml/docker_sort_patches_by_label/sort_patches_by_label_latest.sif'
            slurm_queue: 'dgxV100'
        parameters: 
        output_files: 
'''

#================================================================================
#import modules
#--------------------------------------------------------------------------------
import sys
import kronos.pipelineui
from kronos.helpers import make_dir, kill_jobs, flushqueue, make_intermediate_cmd_args, KeywordsManager
from kronos.logger import PipelineLogger, LogWarnErr, LogInfo
import ruffus
from kronos.kronos_version import kronos_version
from datetime import datetime
import os
from multiprocessing import Queue
import subprocess

#================================================================================
#argument validation
#--------------------------------------------------------------------------------
job_ids = Queue()
job_rcs = Queue()
rc = 0
args = kronos.pipelineui.args

args.components_dir = os.path.abspath(args.components_dir)

if args.pipeline_name is None:
    pipeline_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

else:
    pipeline_name = args.pipeline_name

if args.run_id is None:
    run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

else:
    run_id = args.run_id

make_dir(args.working_dir)
args.working_dir = os.path.join(args.working_dir, run_id)
make_dir(args.working_dir)

if args.log_file is None:
    log_file = os.path.join(args.working_dir, '_'.join([pipeline_name, run_id]) + '.log')

else:
    log_file = os.path.join(args.working_dir, '_'.join([args.log_file, run_id]) + '.log')

#================================================================================
#make a copy of the config file
#--------------------------------------------------------------------------------
cfile = os.path.join(args.working_dir, pipeline_name + '_' + run_id + '.yaml')
with open(cfile, 'w') as cf:
    cf.write(config_file_content)

#================================================================================
#logger initialization
#--------------------------------------------------------------------------------
pl = PipelineLogger()
l = pl.get_logger(pipeline_name, log_file)
pl.log_pipeline_header(l, args, pipeline_name, run_id, kronos_version)

args = vars(kronos.pipelineui.args)

#================================================================================
#ruffus pipeline
#--------------------------------------------------------------------------------
@ruffus.follows()
@LogWarnErr(l)
@LogInfo(l)
def task_0(pipeline_name='__shared__only___preprocess_endometrial'):
    sample_id = '__shared__only__'
    intermediate_path = os.path.join(os.path.dirname(sys.argv[0]),'intermediate_pipeline_scripts')
    pipeline_script = '{0}/{1}.py'.format(intermediate_path, pipeline_name)

    args['pipeline_name'] = pipeline_name
    args['run_id'] = run_id
    args['sample_id'] = sample_id
    args['log_file'] = log_file

    km = KeywordsManager(pipeline_name, run_id, sample_id, args['working_dir'])
    old_script_content = open(pipeline_script, 'r').read()
    new_script_content = km.replace_keywords(old_script_content)
    f = open(pipeline_script, 'w')
    f.write(new_script_content)
    f.close()

    cmd = '{0} {1}'.format(args['python_installation'], pipeline_script)
    cmd_args = make_intermediate_cmd_args(args)
    cmd = cmd + ' ' + ' '.join(cmd_args)
    print 'running __shared__only___preprocess_endometrial pipeline with command: %s' % (cmd)

    proc = subprocess.Popen(cmd, shell=True)
    job_ids.put(proc.pid)
    try:
        cmdout, cmderr = proc.communicate()
        job_rcs.put(proc.returncode)
    except:
        cmd = 'kill %s' % (proc.pid)
        os.system(cmd)
    finally:
        print '__shared__only___preprocess_endometrial pipeline finished with exit code %s' % (proc.returncode)

    if cmdout is not None:
        print >> sys.stdout, cmdout

    if cmderr is not None:
        print >> sys.stderr, cmderr

#================================================================================
#running pipeline
#--------------------------------------------------------------------------------
try:
    ruffus.pipeline_run([task_0], multiprocess=args['num_pipelines'])
    lrc = flushqueue(job_rcs)
    if len(lrc) == len(filter(lambda x: x == 99, lrc)):
        print 'pipeline successfully stopped by breakpoints.'
        rc = 99
    elif not all(rc == 0 for rc in lrc):
        rc = 98

except:
    rc = -1
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print >> sys.stderr, '{0} pipeline failed due to error: {1}, {2}'.format(pipeline_name, exc_type, exc_obj)
    kill_jobs(job_ids)

finally:
    pl.log_pipeline_footer(l)
    pl.log_info(l, '{0} pipeline finished with exit code {1}. Please check the logs.'.format(pipeline_name, rc))
    sys.exit(rc)

