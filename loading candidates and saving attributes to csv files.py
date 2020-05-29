# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:22:59 2020

@author: Emanuel Marques Lourenco
"""
from phcx import Candidate
import os
import numpy as np
import csv


def getCandidates(directory):
    """
    Gets candidates from files

    Parameters
    ----------
    :param directory: directory of the files to be extracted from.

    Returns
    ----------
    :return: Return a list containing the names of the entries in the directory
    given by path.
    """
    candidates = os.listdir(directory)
    return candidates

if __name__ == '__main__':

    directory, fname = os.path.split(
            os.path.abspath(__file__)
            )

    pulsarsDirectory = directory + '\pulsars'
    pulsars = getCandidates(pulsarsDirectory)
    
    #Creating arrays to store the different attributes from the datasets
    pulsar_profiles = [] 
    pulsar_subints = []
    pulsar_snrvalues = []
    pulsar_dmvalues = []
    nonpulsarsDirectory = directory + '\\nonpulsars'
    nonpulsars = getCandidates(nonpulsarsDirectory)
    nonpulsar_profiles = []
    nonpulsar_subints = []
    nonpulsar_snrvalues = []
    nonpulsar_dmvalues = []
    
    for i in pulsars:
        cand = Candidate(os.path.join(pulsarsDirectory,i))
        pulsar_profiles.append(cand.profile)
        """flattenning the matrix of sub integrations by using its mean values
        subints attribute of the candidates contain the sub-integrations (phase-time)
        diagram, already normalised to values between 0 and 1"""
        subintmean = cand.subints.mean(0)
        pulsar_subints.append(subintmean)
        pulsar_dmvalues.append(cand.dm_curve[0])
        pulsar_snrvalues.append(cand.dm_curve[1])
    
    for i in nonpulsars:
        cand = Candidate(os.path.join(nonpulsarsDirectory,i))
        nonpulsar_profiles.append(cand.profile)
        #flattenning the matrix of sub integrations
        subintmean = cand.subints.mean(0)
        nonpulsar_subints.append(subintmean)
        nonpulsar_dmvalues.append(cand.dm_curve[0])
        nonpulsar_snrvalues.append(cand.dm_curve[1])

    
    #saving profiles into an csv file
    np.savetxt('pulsar_profiles.csv',pulsar_profiles,delimiter=',')
    np.savetxt('nonpulsar_profiles.csv',nonpulsar_profiles,delimiter=',')
    
    #saving sub integrations mean values into an csv file
    np.savetxt('pulsar_subintegrations.csv',pulsar_subints,delimiter=',')
    np.savetxt('nonpulsar_subintegrations.csv',nonpulsar_subints,delimiter=',')
    
    #savetxt will not work with dm curves as the arrays are not of the same 
    #length, therefore another method will be used

    with open('pulsar_snrvalues.csv', 'w') as output:
       writer = csv.writer(output, lineterminator='\n')
       for line in pulsar_snrvalues:
           writer.writerow(line) 
    with open('nonpulsar_snrvalues.csv', 'w') as output:
       writer = csv.writer(output, lineterminator='\n')
       for line in nonpulsar_snrvalues:
           writer.writerow(line)
            
    with open('pulsar_dmvalues.csv', 'w') as output:
       writer = csv.writer(output, lineterminator='\n')
       for line in pulsar_dmvalues:
           writer.writerow(line) 
    with open('nonpulsar_dmvalues.csv', 'w') as output:
       writer = csv.writer(output, lineterminator='\n')
       for line in nonpulsar_dmvalues:
           writer.writerow(line)
