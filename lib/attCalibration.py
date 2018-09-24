# -*- coding: utf-8 -*-
"""
    ADC calibration

    Copyright (C) August, 2018  Gascard, Thomas
    Author: Gascard Thomas

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
"""

import numpy as np
import sys, os
import time

# Attenuator RUDAT 6000 30 USB
sys.path.insert(0,'./lib/attenuators/')
import rudat_6000_30_usb

class CalibrationATT(object):

    def __init__(self, attenID):

        # Attenuators
        self.aInCryo = rudat_6000_30_usb.rudats[attenID[0]] # In to cryostat
        self.aOutCryo = rudat_6000_30_usb.rudats[attenID[1]] # Out of cryostat

    def setAtten(self, inAtten, outAtten):
        """Set the input and output attenuation levels for a RUDAT MCL-30-6000
            inputs:
                float outAtten: The output attenuation in dB
                float inAtten: The input attenuation in dB"""
        self.aInCryo.set_atten(inAtten)
        self.aOutCryo.set_atten(outAtten)

#        command = "sudo ./set_rudats " + str(inAtten) + ' ' + str(outAtten)
#        os.system(command)
        return

    def readAtten(self):
        """Read the attenuation levels for both channels of a RUDAT MCL-30-6000
           outputs:
                float outAtten
                float inAtten"""

        inAtten = self.aInCryo.get_atten()
        outAtten = self.aOutCryo.get_atten()
#        os.system("sudo ./read_rudats > rudat.log")
#        attens = np.loadtxt('./rudat.log', delimiter = ",")
#        inAtten = attens[0][1]
#        outAtten = attens[1][1]
        return inAtten, outAtten

    def rmsVoltageADC(self, fpga):
        """Get the voltage RMS for the ADC
            inputs:
                object fpga
            outputs:
                float rmsI: RMS voltage I
                float rmsQ: RMS voltage Q
                float crest_factor_I
                float crest_factor_Q
        """

        # FPGA Registers
        fpga.write_int('adc_snap_adc_snap_ctrl', 0)
        time.sleep(0.1)
        fpga.write_int('adc_snap_adc_snap_ctrl', 1)
        time.sleep(0.1)
        fpga.write_int('adc_snap_adc_snap_ctrl', 0)
        time.sleep(0.1)
        fpga.write_int('adc_snap_adc_snap_trig', 0)
        time.sleep(0.1)
        fpga.write_int('adc_snap_adc_snap_trig', 1)
        time.sleep(0.1)
        fpga.write_int('adc_snap_adc_snap_trig', 0)
        time.sleep(0.1)

        adc = (np.fromstring(fpga.read('adc_snap_adc_snap_bram', (2**10)*8), dtype = '>h')).astype('float')
        adc = adc / ((2**15) - 1)
        adc *= 550.
        I = np.hstack(zip(adc[0::4],adc[2::4]))
        Q = np.hstack(zip(adc[1::4],adc[3::4]))

        rmsI = np.round(np.sqrt(np.mean(I**2)),2)
        rmsQ = np.round(np.sqrt(np.mean(Q**2)),2)

        peakI = np.abs(np.max(I))
        peakQ = np.abs(np.max(Q))

        crest_factor_I = np.round(20.*np.log10(peakI/rmsI) ,2)
        crest_factor_Q = np.round(20.*np.log10(peakQ/rmsQ), 2)

        return rmsI, rmsQ, crest_factor_I, crest_factor_Q

    def calibrateADC(self, fpga, target_rms_mv, outAtten, inAtten): # TG: Needs testing
        """Automatically set RUDAT attenuation values to achieve desired ADC rms level
            inputs:
               float target_rms_mv: The target ADC rms voltage level, in mV,
                                    for either I or Q channel
               float outAtten: Starting output attenuation, dB
               float inAtten: Starting input attenuation, dB
        """

        self.setAtten(outAtten, inAtten)
        print "Start atten:", outAtten, inAtten
        rmsI, rmsQ, __, __ = self.rmsVoltageADC(fpga)
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
                self.setAtten(outAtten, inAtten)
                rmsI, rmsQ, __, __ = self.rmsVoltageADC(fpga)
                avg_rms = (rmsI + rmsQ)/2.
                outA, inA = self.readAtten()
                print "Output Att: " + str(outA) + ", Input Att: " + str(inA)
        if avg_rms_0 > target_rms_mv:
            avg_rms = avg_rms_0
            while avg_rms > target_rms_mv:
                time.sleep(0.1)
                if outAtten < 30:
                    outAtten += 1
                else:
                    inAtten += 1
                if (inAtten >= 30) and (outAtten >= 30):
                    break
                self.setAtten(outAtten, inAtten)
                rmsI, rmsQ, __, __ = self.rmsVoltageADC(fpga)
                avg_rms = (rmsI + rmsQ)/2.
                outA, inA = self.readAtten()
                print "Output Att: " + str(outA) + ", Input Att: " + str(inA)
        new_out, new_in = self.readAtten()
        print
        print "Final atten:", new_out, new_in
        print "Current RMS:", avg_rms, "mV"
        print
        return
