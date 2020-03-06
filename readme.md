### WSI Starter Pipeline ###

```
Development information

date created : 14 Feb 2020
last update  : 14 Feb 2020
Developer    : Mitchell Nursey (mnursey@bccrc.ca)
Version      : 1.0
```

### 1. Getting Started ###

#### Components ####

This pipeline only uses the [Kronos Docker wrapper component](https://svn.bcgsc.ca/bitbucket/projects/MLOVCA/repos/kronos_component_docker/browse).

#### Docker Images ####

The docker images used in this pipeline and are all managed by the Kronos Docker wrapper component.

### 4. Example Command ###
kronos run -c *(LOCATION OF KRONOS DOCKER WRAPPER COMPONENT)* -i $PWD/input.txt -s $PWD/setup.txt -y $PWD/*(WHICH SETUP TO USE)*.yaml --no_prefix
