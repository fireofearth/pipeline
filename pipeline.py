#================================================================================
#config file content
#--------------------------------------------------------------------------------
config_file_content = '''
__PIPELINE_INFO__: 
    date_last_updated: 
    name: 'Preprocessing Pipeline'
    data_type: 
    author: 'mnursey'
    Kronos_version: '2.3.0'
    host_cluster: 
    version: '0.00.1'
    input_type: 
    date_created: '2019-05-07'
    output_type: 
__GENERAL__: 
    singularity: 'singularity'
__SHARED__: 
    anno_loc: 
    patch_loc: 
    img_loc: 
__SAMPLES__: 
    # sample_id:
    #    param1: value1
    #    param2: value2
    SET_0: 
        sort_loc: '/projects/ovcare/classification/1024_Sorted'
        patch_loc: '/projects/ovcare/classification/1024_Patches'
        aug_loc: '/projects/ovcare/classification/1024_Aug'
        label_file_loc: '/projects/ovcare/classification/labels.txt'
AUGMENT_DATA: 
    run: 
        interval_file: 
        requirements: 
            singularity: 
        boilerplate: 
        use_cluster: False
        num_cpus: 1
        parallel_run: False
        merge: True
        add_breakpoint: False
        memory: '5G'
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
            patch_location: "('__SAMPLES__','patch_loc')"
            augmented_location: "('__SAMPLES__','aug_loc')"
            singularity_image: '/projects/ovcare/classification/src/docker_augment_data/augmentdata-latest.simg'
        parameters: 
        output_files: 
SORT_PATCHES: 
    run: 
        interval_file: 
        requirements: 
            singularity: 
        boilerplate: 
        use_cluster: False
        num_cpus: 1
        parallel_run: False
        merge: True
        add_breakpoint: False
        memory: '5G'
        forced_dependencies: ['AUGMENT_DATA']
        env_vars: 
        parallel_params: []
    reserved: 
        # do not change this section.
        seed_version: '0.00.1'
        component_version: '0.00.1'
        component_name: 'kronos_component_singularity'
    component: 
        input_files: 
            sorted_location: "('__SAMPLES__','sort_loc')"
            labels_file_location: "('__SAMPLES__','label_file_loc')"
            patch_location: "('__SAMPLES__','aug_loc')"
            singularity_image: '/projects/ovcare/classification/src/docker_sort_patches_by_label/sortpatchesbylabel.simg'
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
def task_0(pipeline_name='SET_0_pipeline'):
    sample_id = 'SET_0'
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
    print 'running SET_0_pipeline pipeline with command: %s' % (cmd)

    proc = subprocess.Popen(cmd, shell=True)
    job_ids.put(proc.pid)
    try:
        cmdout, cmderr = proc.communicate()
        job_rcs.put(proc.returncode)
    except:
        cmd = 'kill %s' % (proc.pid)
        os.system(cmd)
    finally:
        print 'SET_0_pipeline pipeline finished with exit code %s' % (proc.returncode)

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

