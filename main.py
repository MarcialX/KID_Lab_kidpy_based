# The homodyne system for AIG lab
#
# Copyright (C) November, 2018  Becerril, Marcial <mbecerrilt@inaoep.mx>
# Author: Becerril, Marcial <mbecerrilt@inaoep.mx> based in the codes of
# Sam Gordon <sbgordo1@asu.edu>, Sam Rowe and Thomas Gascard.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import time
import numpy as np
import sys, os
import struct
from socket import *

import casperfpga
from roachInterface import roachInterface
from gbeConfig import roachDownlink

import matplotlib.pyplot as plt
from scipy import signal, ndimage, fftpack
import pygetdata as gd

import find_kids_interactive as fk
import targplot

sys.path.insert(0,'./lib/synth/')
import synthclass

plt.ion()

# Load general settings
gc = np.loadtxt("./config/general_config", dtype = "str")

# Load list of firmware registers (note: must manually update for different versions)
regs = np.loadtxt("./config/firmware_registers", dtype = "str")

# Paths of firmware and directories to save data
firmware = gc[np.where(gc == 'FIRMWARE_FILE')[0][0]][1]
vna_savepath = gc[np.where(gc == 'VNA_SAVEPATH')[0][0]][1]
targ_savepath = gc[np.where(gc == 'TARG_SAVEPATH')[0][0]][1]
dirfile_savepath = gc[np.where(gc == 'DIRFILE_SAVEPATH')[0][0]][1]

# UDP packet
buf_size = int(gc[np.where(gc == 'buf_size')[0][0]][1])
header_len = int(gc[np.where(gc == 'header_len')[0][0]][1])

# About the ROACH
roach_ip = gc[np.where(gc == 'roach_ppc_ip')[0][0]][1]

# Windfreak Synthesizer params
synthID = gc[np.where(gc == 'synthID')[0][0]][1]
clkFreq = np.float(gc[np.where(gc == 'clkFreq')[0][0]][1])
clkPow = np.float(gc[np.where(gc == 'clkPow')[0][0]][1])
LOFreq = np.float(gc[np.where(gc == 'LOFreq')[0][0]][1])
LOPow = np.float(gc[np.where(gc == 'LOPow')[0][0]][1])
center_freq = np.float(gc[np.where(gc == 'center_freq')[0][0]][1])

# Optional test frequencies
test_freq = np.float(gc[np.where(gc == 'test_freq')[0][0]][1])
test_freq = np.array([test_freq])
freq_list = gc[np.where(gc == 'freq_list')[0][0]][1]

# Parameters for resonator search
smoothing_scale = np.float(gc[np.where(gc == 'smoothing_scale')[0][0]][1])
peak_threshold = np.float(gc[np.where(gc == 'peak_threshold')[0][0]][1])
spacing_threshold  = np.float(gc[np.where(gc == 'spacing_threshold')[0][0]][1])

# Menu options for terminal interface
cap_0 = '\n\t\033[95mROACH Readout system. Starting functions\033[95m'
cap_1 = '\n\t\033[95mHomodyne system\033[95m'
cap_2 = '\n\t\033[95mVNA functions\033[95m'

captions = [cap_0,cap_1,cap_2]
main_opts= ['Test connection to ROACH',\
            'Upload firmware',\
            'Initialize system & UDP conn',\
            'Write test comb (single or multitone)',\
            'Write stored comb',\
            'Apply inverse transfer function',\
            'Calibrate ADC V_rms',\
            'Get system state',\
            'Test GbE downlink',\

            'Load resonance frequencies',\
            'Low resolution sweep',\
            'High resolution sweep',\

            'Print packet info to screen (UDP)',\
            'VNA sweep and plot','Locate freqs from VNA sweep',\
            'Write found freqs',\
            'Target sweep and plot',\
            'Plot channel phase PSD (quick look)',\
            'Save dirfile for range of chan',\
	        'Execute a script',\
            'Exit']

def testConn(fpga):
    """Tests the link to Roach2 PPC, using return from getFPGA()
        inputs:
            casperfpga object fpga: The fpga object
        outputs: the fpga object"""
    if not fpga:
        try:
            fpga = casperfpga.CasperFpga(roach_ip, timeout = 3.)
        except RuntimeError:
            print "\nNo connection to ROACH. If booting, wait 30 seconds and retry. Otherwise, check gc config."
    return fpga

def writeVnaComb(cw = False):
    # Roach PPC object
    fpga = getFPGA()
    if not fpga:
        print "\nROACH link is down"
        return
    # Roach interface
    ri = roachInterface(fpga, gc, regs, None)
    try:
        if cw:
            ri.freq_comb = test_freq
        else:
            ri.makeFreqComb()
            if (len(ri.freq_comb) > 400):
                fpga.write_int(regs[np.where(regs == 'fft_shift_reg')[0][0]][1], 2**5 -1)
                time.sleep(0.1)
            else:
                fpga.write_int(regs[np.where(regs == 'fft_shift_reg')[0][0]][1], 2**9 -1)
                time.sleep(0.1)
            ri.upconvert = np.sort(((ri.freq_comb + (center_freq)*1.0e6))/1.0e6)
            print "RF tones =", ri.upconvert
            ri.writeQDR(ri.freq_comb, transfunc = False)
            np.save("last_freq_comb.npy", ri.freq_comb)

            if not (fpga.read_int(regs[np.where(regs == 'dds_shift_reg')[0][0]][1])):
                if regs[np.where(regs == 'DDC_mixerout_bram_reg')[0][0]][1] in fpga.listdev():
                    shift = ri.return_shift(0)
                    if (shift < 0):
                        print "\nError finding dds shift: Try writing full frequency comb (N = 1000), or single test frequency. Then try again"
                        return
                    else:
                        fpga.write_int(regs[np.where(regs == 'dds_shift_reg')[0][0]][1], shift)
                        print "Wrote DDS shift (" + str(shift) + ")"
                else:
                    fpga.write_int(regs[np.where(regs == 'dds_shift_reg')[0][0]][1], ri.dds_shift)
    except KeyboardInterrupt:
        return
    return

# Needs testing
def calibrateADC(target_rms_mv, outAtten, inAtten):
    """Automatically set RUDAT attenuation values to achieve desired ADC rms level
       inputs:
           float target_rms_mv: The target ADC rms voltage level, in mV,
                                for either I or Q channel
           float outAtten: Starting output attenuation, dB
           float inAtten: Starting input attenuation, dB"""
    #setAtten(outAtten, inAtten)
    fpga = getFPGA()
    valon=None
    ri = roachInterface(fpga, gc, regs, valon)
    print "Start atten:", outAtten, inAtten
    rmsI, rmsQ, __, __ = ri.rmsVoltageADC()
    avg_rms_0 = (rmsI + rmsQ)/2.
    print "Target RMS:", target_rms_mv, "mV"
    print "Current RMS:", avg_rms_0, "mV"
    if avg_rms_0 < target_rms_mv:
        avg_rms = avg_rms_0
        while avg_rms < target_rms_mv:
            time.sleep(0.1)
            if inAtten > 1:
                inAtten -= 1
            else:
                outAtten -= 1
            if (inAtten == 1) and (outAtten == 1):
                break
            #setAtten(outAtten, inAtten)
            rmsI, rmsQ, __, __ = ri.rmsVoltageADC()
            avg_rms = (rmsI + rmsQ)/2.
            outA, inA = 0,0 #readAtten()
            print outA, inA
    if avg_rms_0 > target_rms_mv:
        avg_rms = avg_rms_0
        while avg_rms > target_rms_mv:
            time.sleep(0.1)
            if outAtten < 30:
                outAtten += 1
            else:
                inAtten += 1
            if (inAtten > 30) and (outAtten > 30):
                break
            #setAtten(outAtten, inAtten)
            rmsI, rmsQ, __, __ = ri.rmsVoltageADC()
            avg_rms = (rmsI + rmsQ)/2.
            outA, inA = 0,0 #readAtten()
            print outA, inA
    new_out, new_in = 0,0 #readAtten()
    print
    print "Final atten:", new_out, new_in
    print "Current RMS:", avg_rms, "mV"
    return

def getSystemState(fpga, ri, udp, valon):
    """Displays current firmware configuration
       inputs:
           casperfpga object fpga
           roachInterface object ri
           gbeConfig object udp
           valon synth object valon"""
    print
    print "***Current system state***"
    print "DDS shift:", fpga.read_int(regs[np.where(regs == 'dds_shift_reg')[0][0]][1])
    print "FFT shift:", fpga.read_int(regs[np.where(regs == 'fft_shift_reg')[0][0]][1])
    print "Number of tones:", fpga.read_int(regs[np.where(regs == 'read_comb_len_reg')[0][0]][1])
    print "QDR Cal status:", fpga.read_int(regs[np.where(regs == 'read_qdr_status_reg')[0][0]][1])
    print
    print "***Data downlink***"
    print "Stream status: ", fpga.read_int(regs[np.where(regs == 'read_stream_status_reg')[0][0]][1])
    print "Data rate: ", ri.accum_freq, "Hz", ", " + str(np.round(buf_size * ri.accum_freq / 1.0e6, 2)) + " MB/s"
    print "UDP source IP,port:", inet_ntoa(struct.pack(">i", fpga.read_int(regs[np.where(regs == 'udp_srcip_reg')[0][0]][1]))),":", fpga.read_int(regs[np.where(regs == 'udp_srcport_reg')[0][0]][1])
    print "UDP dest IP,port:", inet_ntoa(struct.pack(">i", fpga.read_int(regs[np.where(regs == 'udp_destip_reg')[0][0]][1]))),":", fpga.read_int(regs[np.where(regs == 'udp_destport_reg')[0][0]][1])
    print
    print "***ADC and attenuator levels***"
    inAtten, outAtten = 0,0 #readAtten()
    rmsI, rmsQ, crest_factor_I, crest_factor_Q = ri.rmsVoltageADC()
    print "out atten:", outAtten, "dB"
    print "in atten:", inAtten, "dB"
    print "ADC V_rms (I,Q):", rmsI, "mV", rmsQ, "mV"
    print "Crest factor (I,Q):", crest_factor_I, "dB", crest_factor_Q, "dB"
    print
    print "***Synthesizer " + str(synthID) + "***"
    print "LO center freq:", center_freq, "MHz"
    print "LO freq:", LOFreq, "MHz"
    print "LO power:", LOPow, "dB"
    print "Frequency Clock", clkFreq, "MHz"
    print "Power clock:", clkPow, "dB"
    return

def vnaSweep(ri, udp, valon):
    """Does a wideband sweep of the RF band, saves data in vna_savepath
       as .npy files
       inputs:
           roachInterface object ri
           gbeConfig object udp
           valon synth object valon
           bool write: Write test comb before sweeping?
           Navg = Number of data points to average at each sweep step"""

    Navg = np.int(gc[np.where(gc == 'Navg')[0][0]][1])
    if not os.path.exists(vna_savepath):
        os.makedirs(vna_savepath)
    sweep_dir = vna_savepath + '/' + \
       str(int(time.time())) + '-' + time.strftime('%b-%d-%Y-%H-%M-%S') + '.dir'

    os.mkdir(sweep_dir)
    np.save("./last_vna_dir.npy", sweep_dir)

    print sweep_dir
    #valon.set_frequency(LO, center_freq/1.0e6)
    span = ri.pos_delta

    print "Sweep Span =", 2*np.round(ri.pos_delta,2), "Hz"
    start = center_freq*1.0e6 - (span)
    stop = center_freq*1.0e6 + (span)
    sweep_freqs = np.arange(start, stop, lo_step)
    sweep_freqs = np.round(sweep_freqs/lo_step)*lo_step

    if not np.size(ri.freq_comb):
        ri.makeFreqComb()
    np.save(sweep_dir + '/bb_freqs.npy', ri.freq_comb)
    np.save(sweep_dir + '/sweep_freqs.npy', sweep_freqs)
    Nchan = len(ri.freq_comb)

    if not Nchan:
        Nchan = fpga.read_int(regs[np.where(regs == 'read_comb_len_reg')[0][0]][1])

    for freq in sweep_freqs:
        print 'LO freq =', freq/1.0e6
        #valon.set_frequency(LO, freq/1.0e6)
        #print "LO freq =", valon.get_frequency(LO)
        #time.sleep(0.1)
        udp.saveSweepData(Navg, sweep_dir, freq, Nchan,skip_packets = 10)
        #time.sleep(0.1)
    #valon.set_frequency(LO, center_freq) # LO
    return

def plotVNASweep(path):
    plt.figure()
    Is, Qs = openStoredSweep(path)
    sweep_freqs = np.load(path + '/sweep_freqs.npy')
    bb_freqs = np.load(path + '/bb_freqs.npy')
    rf_freqs = np.zeros((len(bb_freqs),len(sweep_freqs)))

    for chan in range(len(bb_freqs)):
        rf_freqs[chan] = (sweep_freqs + bb_freqs[chan])/1.0e6

    Q = np.reshape(np.transpose(Qs),(len(Qs[0])*len(sweep_freqs)))
    I = np.reshape(np.transpose(Is),(len(Is[0])*len(sweep_freqs)))
    mag = np.sqrt(I**2 + Q**2)
    mag = 20*np.log10(mag/np.max(mag))
    mag = np.concatenate((mag[len(mag)/2:],mag[:len(mag)/2]))
    rf_freqs = np.hstack(rf_freqs)
    rf_freqs = np.concatenate((rf_freqs[len(rf_freqs)/2:],rf_freqs[:len(rf_freqs)/2]))

    plt.plot(rf_freqs, mag)
    plt.title(path, size = 16)
    plt.xlabel('frequency (MHz)', size = 16)
    plt.ylabel('dB', size = 16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(path,'vna_sweep.png'), dpi = 100, bbox_inches = 'tight')

    return

def openStoredSweep(savepath):
    """Opens sweep data
       inputs:
           char savepath: The absolute path where sweep data is saved
       ouputs:
           numpy array Is: The I values
           numpy array Qs: The Q values"""
    files = sorted(os.listdir(savepath))
    I_list, Q_list = [], []
    for filename in files:
        if filename.startswith('I'):
            I_list.append(os.path.join(savepath, filename))
        if filename.startswith('Q'):
            Q_list.append(os.path.join(savepath, filename))
    Is = np.array([np.load(filename) for filename in I_list])
    Qs = np.array([np.load(filename) for filename in Q_list])
    return Is, Qs

def getFPGA():
    """Returns a casperfpga object of the Roach2"""
    try:
        fpga = casperfpga.CasperFpga(roach_ip, timeout = 120.)
    except RuntimeError:
        print "\nNo connection to ROACH. If booting, wait 30 seconds and retry. Otherwise, check gc config."
    return fpga

def loadResFreq(ri, tonesListPath):
    """Generates a frequency comb for the DAC or DDS look-up-tables.
    Parameters:
        object ri: Roach Controller
        string tonesList: Path and filename of the list of resonator's tones.
    Return:
        floats[] resFreqs: List of frequencies corresponding to the resonnator's tones,
        str[] resID: List of associated resonnator IDs, to be compared to the actual KIDs ID from design
    """

    """
        Tones List text files:
            'Date'
            'Time'
            'Run': ID of the experiment
            'Array' :
            'Chip' : ID of the KID
            'ResID' : ID of each KID. Normally the number
            #   'ResFreqSweep' : Resonance Freq from Sweep [Hz]
            'ResFreqDesign' : Resonance Freq from Design [Hz]
            'Temp' : Temperature at which the sweep has been conducted [deg C]
            'PowInCryo'
            'PowOutCryo'
            #   'QSweep'
    """

    run = KID_Header[0]
    chip = KID_Header[1]

    # Archivo de texto
    if tonesListPath.endswith('.txt'):
        pattern = re.compile(r"[0-9]{4,12}")

        regexFreqs = []
        resFreqs = []
        with open(tonesListPath, "rt") as in_file:
            for line in in_file:
                if pattern.search(line) != None :
                    regexFreqs.append(re.findall(pattern, line))

        resID = range(len(regexFreqs))
        for i in resID:
            resFreqs.append(int(regexFreqs[i][0]))

        tonesList = pd.DataFrame({'Date' : '',\
                                  'Time' : '',\
                                  'Run' : run,\
                                  'Array' : '',\
                                  'Chip' : chip,\
                                  'ResID' : resID,\
                                  'ResFreqSweep' : resFreqs,\
                                  'Temp' : '',\
                                  'PowInCryo' : '',\
                                  'PowOutCryo' : '',\
                                  'QSweep' : ''})

    # Archivo fits
    if vnaSweepPath.endswith('.fits'):
        sweepfile = importdata.importData(vnaSweepPath)
        h, d = sweepfile.extractHDU()

        fiq = [d['Frequency'], d['RawI'], d['RawQ']]
        resFreqs = ri.find_kids_vna_dirfile(fiq)

        estQr = []
        for i in range(0, len(resFreqs)):
            fr = resFreqs[i]
            estQr.append(km.estimateQr(fr, fiq[0], fiq[1], fiq[2]))

        tonesList = pd.DataFrame({'Date' : h['DATE'],\
                                  'Time' : h['TIME'],\
                                  'Run' : run,\
                                  'Array' : h['DUT'],\
                                  'Chip' : chip,\
                                  'ResID' : resID,\
                                  'ResFreqSweep' : resFreqs,\
                                  'Temp' : h['SAMPLETE'],\
                                  'PowInCryo' : h['INPUTAT'],\
                                  'PowOutCryo' : h['OUTPUTAT'],\
                                  'QSweep' : estQr})

    return resID, resFreqs, tonesList

def lowResSweep(ri, resFreqs, freqSpan, numPoints): # ??? savePath

    """Low resolution sweep to find where the resonators have moved,

    Parameters:
        float[] resFreq : resonance frequency loaded from the toneslist with the loadResFrequencies function,
        float freqSpan: full frequency bandwith around the resonnant frequency, in Hz,
        int numPoints: sweep resolution as a number of points,
        string savePath: path, non including filename, where the data will be saved.

    Return:
        f_lr, i_lr, q_lr (array of arrays of floats): I / Q as a function of frequency for the low resolution sweep

    Notes for later version:
        Implement averaging? targetPow?

    """
    # Low resolution sweep
    center_freq = (np.max(resFreqs) + np.min(resFreqs))/2.
    bb_freqs = resFreqs - center_freq

    ri.writeQDR(bb_freqs)
    save_path = './lr_lo_sweeps'

    print "Starting Low Resolution Sweep"
    f_lr, i_lr, q_lr = ri.sweep_lo_dirfile(center_freq, freqSpan, save_path, numPoints, bb_freqs)

    return f_lr, i_lr, q_lr

def lowResSweep(ri, resFreqs, freqSpan, numPoints): # ??? savePath

    """Low resolution sweep to find where the resonators have moved,

    Parameters:
        float[] resFreq : resonance frequency loaded from the toneslist with the loadResFrequencies function,
        float freqSpan: full frequency bandwith around the resonnant frequency, in Hz,
        int numPoints: sweep resolution as a number of points,
        string savePath: path, non including filename, where the data will be saved.

    Return:
        f_lr, i_lr, q_lr (array of arrays of floats): I / Q as a function of frequency for the low resolution sweep

    Notes for later version:
        Implement averaging? targetPow?

    """
    # High resolution sweep
    freqSpan = freqSpan /2
    numPoints = 2*numPoints
    save_path = './hr_lo_sweeps'

    print "Start High Resolution Sweep"
    f_hr, i_hr, q_hr = ri.sweep_lo_dirfile(center_freq, freqSpan, save_path, numPoints, bb_freqs)

    return f_hr, i_hr, q_hr

def menu(captions, options):
    """Creates menu for terminal interface
       inputs:
           list captions: List of menu captions
           list options: List of menu options
       outputs:
           int opt: Integer corresponding to menu option chosen by user"""

    cnt_cap = 0
    tar_cap = [0,9,12,len(captions)-1]
    opt = -1
    while opt >= len(options) or opt < 0:
        for i in range(len(options)):
            if i == tar_cap[cnt_cap]:
                print '\t' + captions[cnt_cap] + '\n'
                cnt_cap += 1
            print '\t' +  '\033[32m' + str(i) + ' ..... ' '\033[0m' +  options[i] + '\n'
        try:
            opt = input('Task: ')
        except:
            print '\n\t\033[93mOption not valid. Try again.\033[93m'
    return opt

def main_opt(fpga, ri, udp, valon, upload_status):
    """Creates terminal interface
       inputs:
           casperfpga object fpga
           roachInterface object ri
           gbeConfig object udp
           valon synth object valon
           int upload_status: Integer indicating whether or not firmware is uploaded
        outputs:
          int  upload_status"""
    while 1:
        if not fpga:
            print '\n\t\033[93mROACH link is down: Check PPC IP & Network Config\033[93m'
        else:
            print '\n\t\033[92mROACH link is up\033[92m'
        if not upload_status:
            print '\n\t\033[93mNo firmware onboard. If ROACH link is up, try upload option\033[93m'
        else:
            print '\n\t\033[92mFirmware uploaded\033[92m'
        opt = menu(captions,main_opts)

        if opt == 0:
            result = testConn(fpga)
            if not result:
                break
            else:
                fpga = result
                print "\n Connection is up"

        if opt == 1:
            if not fpga:
                print "\nROACH link is down"
                break
            if (ri.uploadfpg() < 0):
                print "\nFirmware upload failed"
            else:
                upload_status = 1

        if opt == 2:
            if not fpga:
                print "\nROACH link is down"
                break
            os.system('clear')

            # Initializing Synthesizer Windfreak
            synthRF = synthclass.Synthesizer(synthID)

            fpga.write_int(regs[np.where(regs == 'accum_len_reg')[0][0]][1], ri.accum_len - 1)
            time.sleep(0.1)
            fpga.write_int(regs[np.where(regs == 'dds_shift_reg')[0][0]][1], int(gc[np.where(gc == 'dds_shift')[0][0]][1]))
            time.sleep(0.1)

            # QDR Calibration
            if (ri.qdrCal() < 0):
                print '\033[93mQDR calibration failed... Check FPGA clock source\033[93m'
                break
            else:
                fpga.write_int(regs[np.where(regs == 'write_qdr_status_reg')[0][0]][1], 1)
                print "QDR calibration completed!"
            time.sleep(0.1)

            # UDP Configuration
            try:
                udp.configDownlink()
            except AttributeError:
                print "UDP Downlink could not be configured. Check ROACH connection."
                break

        if opt == 3:
             prompt = raw_input('Full test comb? y/n ')
             if prompt == 'y':
                 writeVnaComb()
             else:
                 writeVnaComb(cw = True)

        if opt == 4:
            if not fpga:
                print "\nROACH link is down"
                break
            try:
                freq_comb = np.load(freq_list)
                freq_comb = freq_comb[freq_comb != 0]
                freq_comb = np.roll(freq_comb, - np.argmin(np.abs(freq_comb)) - 1)
                ri.freq_comb = freq_comb
                ri.upconvert = np.sort(((ri.freq_comb + (ri.center_freq)*1.0e6))/1.0e6)
                print "RF tones =", ri.upconvert
                if len(ri.freq_comb) > 400:
                    fpga.write_int(regs[np.where(regs == 'fft_shift_reg')[0][0]][1], 2**5 -1)
                    time.sleep(0.1)
                else:
                    fpga.write_int(regs[np.where(regs == 'fft_shift_reg')[0][0]][1], 2**9 -1)
                    time.sleep(0.1)
                ri.writeQDR(ri.freq_comb)
                #setAtten(27, 17)
                np.save("last_freq_comb.npy", ri.freq_comb)
            except KeyboardInterrupt:
                pass
        if opt == 5:
            if not fpga:
                print "\nROACH link is down"
                break
            if not np.size(ri.freq_comb):
                try:
                    ri.freq_comb = np.load("last_freq_comb.npy")
                except IOError:
                   print "\nFirst need to write a frequency comb with length > 1"
                   break
            try:
                ri.writeQDR(ri.freq_comb, transfunc = True)
                fpga.write_int(regs[np.where(regs == 'write_comb_len_reg')[0][0]][1], len(ri.freq_comb))
            except ValueError:
                print "\nClose Accumulator snap plot before calculating transfer function"
        if opt == 6:
            if not fpga:
                print "\nROACH link is down"
                break
            try:
                calibrateADC(83., 20, 20)
            except KeyboardInterrupt:
                pass
        if opt == 7:
            if not fpga:
                print "\nROACH link is down"
                break
            getSystemState(fpga, ri, udp, valon)
        if opt == 8:
            if not fpga:
                print "\nROACH link is down"
                break
            if (udp.testDownlink(5) < 0):
                print "Error receiving data. Check ethernet configuration."
            else:
                print "OK"
                fpga.write_int(regs[np.where(regs == 'write_stream_status_reg')[0][0]][1], 1)

        if opt == 9:
            if not fpga:
                print "\nROACH link is down"
                break
            print "Load resonance frequencies"

        if opt == 10:
            if not fpga:
                print "\nROACH link is down"
                break
            print "Low resolution sweep"

        if opt == 11:
            if not fpga:
                print "\nROACH link is down"
                break
            print "High resolution sweep"

        if opt == 12:
            if not fpga:
                print "\nROACH link is down"
                break
            time_interval = input('\nNumber of seconds to stream? ' )
            chan = input('chan = ? ')
            try:
                udp.printChanInfo(chan, time_interval)
            except KeyboardInterrupt:
                pass
        if opt == 13:
            if not fpga:
                print "\nROACH link is down"
                break
            try:
                vnaSweep(ri, udp, None)
                plotVNASweep(str(np.load("last_vna_dir.npy")))
            except KeyboardInterrupt:
                pass
        if opt == 14:
            try:
                path = str(np.load("last_vna_dir.npy"))
                print "Sweep path:", path
                fk.main(path, center_freq, lo_step, smoothing_scale, peak_threshold, spacing_threshold)
                #findFreqs(str(np.load("last_vna_dir.npy")), plot = True)
            except KeyboardInterrupt:
                break
        if opt == 15:
            if not fpga:
                print "\nROACH link is down"
                break
            try:
                freq_comb = np.load(os.path.join(str(np.load('last_vna_dir.npy')), 'bb_targ_freqs.npy'))
                freq_comb = freq_comb[freq_comb != 0]
                freq_comb = np.roll(freq_comb, - np.argmin(np.abs(freq_comb)) - 1)
                ri.freq_comb = freq_comb
                print ri.freq_comb
                #ri.upconvert = np.sort(((ri.freq_comb + (center_freq)*1.0e6))/1.0e6)
                #print "RF tones =", ri.upconvert
                if len(ri.freq_comb) > 400:
                    fpga.write_int(regs[np.where(regs == 'fft_shift_reg')[0][0]][1], 2**5 -1)
                    time.sleep(0.1)
                else:
                    fpga.write_int(regs[np.where(regs == 'fft_shift_reg')[0][0]][1], 2**9 -1)
                    time.sleep(0.1)
                ri.writeQDR(ri.freq_comb)
                #setAtten(27, 17)
                np.save("last_freq_comb.npy", ri.freq_comb)
            except KeyboardInterrupt:
                pass
        if opt == 16:
            if not fpga:
                print "\nROACH link is down"
                break
            try:
                targetSweep(ri, udp, valon)
                plotTargSweep(str(np.load("last_targ_dir.npy")))
            except KeyboardInterrupt:
                pass
        if opt == 17:
            if not fpga:
                print "\nROACH link is down"
                break
            chan = input('Channel number = ? ')
            time_interval = input('Time interval (s) ? ')
            try:
                plotPhasePSD(chan, udp, ri, time_interval)
            except KeyboardInterrupt:
                pass
        if opt == 18:
            if not fpga:
                print "\nROACH link is down"
                break
            time_interval = input('Time interval (s) ? ')
            try:
                #udp.saveDirfile_chanRange(time_interval)
                udp.saveDirfile_chanRangeIQ(time_interval)
                #udp.saveDirfile_adcIQ(time_interval)
            except KeyboardInterrupt:
                pass
        if opt == 19:
            if not fpga:
                print "\nROACH link is down"
                break
            try:
		prompt = raw_input("what is the filename of the script to be executed: ")
		execfile("./scripts/"+prompt)
            except KeyboardInterrupt:
                pass
        if opt == 20:
            sys.exit()

        return upload_status

def main():
    """Main function, try to initialize the system the first time, then open the menu"""

    s = None
    try:
        fpga = casperfpga.CasperFpga(roach_ip, timeout = 120.)
        print "\nConnected to: " + roach_ip
    except RuntimeError:
        fpga = None
        print "\nRoach link is down"

    # UDP socket
    s = socket(AF_PACKET, SOCK_RAW, htons(3))

    # Roach interface
    ri = roachInterface(fpga, gc, regs, None)

    # Windfreak synthesizer
    synthRF = synthclass.Synthesizer(synthID)

    # GbE interface
    udp = roachDownlink(ri, fpga, gc, regs, s, ri.accum_freq)
    udp.configSocket()

    os.system('clear')

    valon = None

    while 1:
        try:
            upload_status = 0
            if fpga:
                if fpga.is_running():
                    upload_status = 1
            time.sleep(0.1)
            #fpga = None
            #ri = None
            #udp = None
            upload_status = main_opt(fpga, ri, udp, valon, upload_status)
        except TypeError:
            pass
    return

if __name__ == "__main__":
    main()
