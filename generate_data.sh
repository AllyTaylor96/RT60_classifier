#!/bin/sh
#$ -N generate_data              
#$ -cwd                  
#$ -l h_rt=02:00:00
#$ -pe gpu 1
#$ -l h_vmem=4G
#$ -o /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s1990126_Alasdair_Taylor/job_logs/$JOB_NAME_$JOB_ID.stdout
#$ -e /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s1990126_Alasdair_Taylor/job_logs/$JOB_NAME_$JOB_ID.stderr
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem
# Initialise the environment modules
. /etc/profile.d/modules.sh

module load anaconda
# Load Python
source activate reverb

# Run the program
python ./generate_data.py /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s1990126_Alasdair_Taylor/data/alba_speech /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s1990126_Alasdair_Taylor/data/rt60_classifier_data/IRs /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s1990126_Alasdair_Taylor/data/rt60_classifier_data/specs
