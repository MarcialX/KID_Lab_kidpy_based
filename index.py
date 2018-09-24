#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

#********************************************************************
#*                              KID-LAB                             *
#*                               INAOE                              *
#*                             index.py                             *
#*                        Programa principal                        *
#*                      Marcial Becerril Tapia                      *
#*                 16/septiembre/2018 !Viva México!                 *
#********************************************************************

import os
import os.path
import sys
import time
import numpy as np
from numpy import fft
import struct
from socket import *
from scipy import signal, ndimage, fftpack
import logging

import casperfpga
import pygetdata as gd
#import find_kids_interactive as fk
#import targplot

from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
from IPython.qt.inprocess import QtInProcessKernelManager

from PyQt4 import QtCore, QtGui,uic
from PyQt4.QtCore import *
from PyQt4.QtGui import QPalette,QWidget,QFileDialog,QMessageBox, QTreeWidgetItem, QIcon, QPixmap

import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import(
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)

import re # ??? Regular expression module to read TonesList.txt file, will change in later version

sys.path.insert(0,'./lib/')
import attCalibration
from roachInterface import roachInterface
from gbeConfig import roachDownlink

sys.path.insert(0,'./lib/synth/')
import synthclass

plt.ion()

# Logging
class QTextEditLogger(logging.Handler):
    def __init__(self,parent):
        logging.Handler.__init__(self)
        self.widget = QtGui.QPlainTextEdit(parent)
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)

# Embed IPython terminal
class EmbedIPython(RichIPythonWidget):

    def __init__(self, **kwarg):
        super(RichIPythonWidget, self).__init__()
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel = self.kernel_manager.kernel
        self.kernel.gui = 'qt4'
        self.kernel.shell.push(kwarg)
        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()

# Main Window class
class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)

        # Load of main window GUI
        # The GUI was developed in QT Designer
        self.ui = uic.loadUi("src/gui/main.ui")

        # Full Screen
        self.ui.showMaximized()
        screen = QtGui.QDesktopWidget().screenGeometry()

        # Screen dimensions
        self.size_x = screen.width()
        self.size_y = screen.height()

        self.ui.setWindowFlags(self.ui.windowFlags() | QtCore.Qt.CustomizeWindowHint)
        self.ui.setWindowFlags(self.ui.windowFlags() & ~QtCore.Qt.WindowMaximizeButtonHint)

        self.ui.ctrlFrame.resize(280,self.size_y - 120)
        self.ui.tabPlots.resize(self.size_x - 600, self.size_y - 150)

        self.ui.plotFrame.resize(self.size_x - 620,self.size_y - 210)
        self.ui.plotFrame.setLayout(self.ui.MainPlot)

        self.ui.plotFrame_2.resize(self.size_x - 620,self.size_y - 210)
        self.ui.plotFrame_2.setLayout(self.ui.MainPlot_2)

        self.ui.Terminal.move(self.size_x - 290,self.size_y/2 - 100)
        self.ui.Terminal.resize(self.size_x - 1085, self.size_y/2 - 30)
        self.ui.Terminal.setLayout(self.ui.TermVBox)

        self.ui.loggsFrame.move(self.size_x - 290,10)
        self.ui.loggsFrame.resize(self.size_x - 1085, self.size_y/2 - 120)
        self.ui.loggsFrame.setLayout(self.ui.loggsText)

        # Logging
        logTextBox = QTextEditLogger(self)
        # You can format what is printed to text box
        logTextBox.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(logTextBox)
        # You can control the logging level
        logging.getLogger().setLevel(logging.INFO)
        self.ui.loggsText.addWidget(logTextBox.widget)
        
        # Initial settings
        # Loading General Configuration file

        # Load general settings
        self.gc = {}
        with open("./config/general_config") as f:
            for line in f:
                if line[0] != "#" and line[0] != "\n":
                    (key, val) = line.split()
                    self.gc[key] = val

        logging.info('Loading configuration parameters ...')

        # Load list of firmware registers (note: must manually update for different versions)
        self.regs = {}
        with open("./config/firmware_registers") as f:
            for line in f:
                if line[0] != "#" and line[0] != "\n":
                    (key, val) = line.split()
                    self.regs[key] = val

        logging.info('Loading firmware registers ...')

        # Paths of firmware and directories to save data
        self.firmware = self.gc['FIRMWARE_FILE']
        self.ui.firmEdit.setText(self.firmware)

        self.vna_savepath = self.gc['VNA_SAVEPATH']
        self.targ_savepath = self.gc['TARG_SAVEPATH']
        self.dirfile_savepath = self.gc['DIRFILE_SAVEPATH']
        self.ui.vnaEdit.setText(self.vna_savepath)
        self.ui.tarEdit.setText(self.targ_savepath)
        self.ui.streamEdit.setText(self.dirfile_savepath)

        # UDP packet
        self.buf_size = int(self.gc['buf_size'])
        self.header_len = int(self.gc['header_len'])

        # Ethernet port
        self.eth_port = self.gc['udp_dest_device']
        self.ui.ethEdit.setText(self.eth_port)
        os.system("sudo ip link set " + self.eth_port + " mtu 9000")

        # Source (V6) Data for V6
        self.udp_src_ip =  self.gc['udp_src_ip']
        self.udp_src_mac =  self.gc['udp_src_mac']
        self.udp_src_port =  self.gc['udp_src_port']

        self.ui.ipSrcEdit.setText(self.udp_src_ip)
        self.ui.macSrcEdit.setText(self.udp_src_mac)
        self.ui.portSrcEdit.setText(self.udp_src_port)

        self.dds_shift =  self.gc['dds_shift']

        self.udp_dst_ip =  self.gc['udp_dest_ip']
        self.udp_dst_mac =  self.gc['udp_dest_mac']
        self.udp_dst_port =  self.gc['udp_dst_port']
        self.ui.ipDstEdit.setText(self.udp_dst_ip)
        self.ui.macDstEdit.setText(self.udp_dst_mac)
        self.ui.portDstEdit.setText(self.udp_dst_port)

        # About the ROACH
        self.roach_ip = self.gc['roach_ppc_ip']
        self.ui.roachIPEdit.setText(self.roach_ip)

        # Windfreak Synthesizer params
        self.synthID = self.gc['synthID']
        self.clkFreq = np.float(self.gc['clkFreq'])
        self.clkPow = np.float(self.gc['clkPow'])
        self.LOFreq = np.float(self.gc['LOFreq'])
        self.LOPow = np.float(self.gc['LOPow'])
        self.center_freq = np.float(self.gc['center_freq'])

        self.lo_step = np.float(self.gc['lo_step'])

        self.ui.freqClk.setText(str(self.clkFreq))
        self.ui.powClk.setText(str(self.clkPow))
        self.ui.loFreq.setText(str(self.LOFreq))
        self.ui.loPow.setText(str(self.LOPow))

        # Limits of test comb
        self.min_pos_freq = np.float(self.gc['min_pos_freq'])
        self.max_pos_freq = np.float(self.gc['max_pos_freq'])
        self.min_neg_freq = np.float(self.gc['min_neg_freq'])
        self.max_neg_freq = np.float(self.gc['max_neg_freq'])
        self.symm_offset = np.float(self.gc['symm_offset'])
        self.Nfreq = int(self.gc['Nfreq'])

        self.ui.minPosEdit.setText(str(self.min_pos_freq/1.0e6))
        self.ui.maxPosEdit.setText(str(self.max_pos_freq/1.0e6))
        self.ui.minNegEdit.setText(str(self.min_neg_freq/1.0e6))
        self.ui.maxNegEdit.setText(str(self.max_neg_freq/1.0e6))
        self.ui.offsetEdit.setText(str(self.symm_offset/1.0e6))
        self.ui.nFreqsEdit.setText(str(self.Nfreq))

        # Attenuation
        att_ID_1 = int(self.gc['att_ID_1'])
        att_ID_2 = int(self.gc['att_ID_2'])

        self.attenID = [att_ID_1,  att_ID_2]

        self.ui.attInIDEdit.setText(str(att_ID_1))
        self.ui.attOutIDEdit.setText(str(att_ID_2))

        self.att_In = int(self.gc['attIn'])
        self.att_Out = int(self.gc['attOut'])
        self.target_rms = np.float(self.gc['target_rms_mv'])

        self.ui.attInEdit.setText(str(self.att_In))
        self.ui.attOutEdit.setText(str(self.att_Out))
        self.ui.tarLevelEdit.setText(str(self.target_rms))

        # Optional test frequencies
        self.test_freq = np.float(self.gc['test_freq'])
        self.test_freq = np.array([self.test_freq])
        self.freq_list = self.gc['freq_list']

        # Parameters for resonator search
        self.smoothing_scale = np.float(self.gc['smoothing_scale'])
        self.peak_threshold = np.float(self.gc['peak_threshold'])
        self.spacing_threshold  = np.float(self.gc['spacing_threshold'])

        # VNA Sweep
        self.startVNA = -255.5e6
        self.stopVNA = 255.5e6

        self.ui.centralEdit.setText(str(self.center_freq))
        self.ui.startEdit.setText(str(self.startVNA/1.0e6))
        self.ui.stopEdit.setText(str(self.stopVNA/1.0e6))
        self.ui.stepEdit.setText(str(self.lo_step/1.0e6))
        self.ui.nTonesEdit.setText(str(self.Nfreq))

        # Tool bar
        # ROACH status
        self.ui.actionRoach.triggered.connect(self.roach_connection)
        # ROACH network
        self.ui.actionNetwork.triggered.connect(self.roach_network)
        # Synthesizer
        self.ui.actionSynthesizer.triggered.connect(self.roach_synth)
        # Attenuattors
        self.ui.actionRF_Calibration.triggered.connect(self.roach_atten)
        # QDR Calibration
        self.ui.actionQDR_Calibration.triggered.connect(self.qdr_cal)

        # Buttons
        # Roach
        # Roach Settings
        self.ui.firmDir.mousePressEvent = self.chooseFirmPath
        self.ui.vnaDir.mousePressEvent = self.chooseVNAPath
        self.ui.targDir.mousePressEvent = self.chooseTargPath
        self.ui.streamDir.mousePressEvent = self.chooseStreamPath

        self.ui.upFirmBtn.mousePressEvent = self.upload_firmware
        self.ui.synthBtn.mousePressEvent = self.roach_synth
        self.ui.udpConfBtn.mousePressEvent = self.roach_network
        self.ui.udpTestBtn.mousePressEvent = self.test_udp
        self.ui.attBtn.mousePressEvent = self.roach_atten
        self.ui.writeTestBtn.mousePressEvent = self.write_test_comb

        self.ui.plotSweepBtn.mousePressEvent = self.start_plot_VNA
        self.ui.startSweepBtn.mousePressEvent = self.start_VNA_sweep

        # Iniatialising
        self.statusConn = 0
        self.statusFirm = 0
        self.statusSynth = 0
        self.statusAtt = 0
        self.statusNet = 0

        self.s = None
        self.fpga = None

        try:
            self.fpga = casperfpga.CasperFpga(self.roach_ip, timeout = 100.)
            icon.addPixmap(QPixmap('./src/icon/ok_icon.png'))
            self.ui.actionRoach_Status.setIcon(icon)
            logging.info("Connected to: " + self.roach_ip)
        except:
            self.fpga = None
            self.statusConn = 1
            logging.info("Roach link is down")

        # Check firmware
        if self.fpga:
            logging.info('Firmware is uploaded')
            if self.fpga.is_running():
                self.ui.upFirmBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: green
                                        }""")
            else:
                self.statusFirm = 1
        else:
            self.statusFirm = 1

        # UDP socket
        # Run with root permissions
        try:
            self.s = socket(AF_PACKET, SOCK_RAW, htons(3))
            logging.info('Socket is initialised.')
        except:
            logging.error('Socket is not initialised. Permissions are required')

        # Roach interface
        self.ri = roachInterface(self.fpga, self.gc, self.regs, None)

        # GbE interface
        try:
            self.udp = roachDownlink(self.ri, self.fpga, self.gc, self.regs, self.s, self.ri.accum_freq)
            self.udp.configSocket()
            logging.info('UDP configuration done.')
        except:
            logging.error("UDP connection couldn't be initialised.")            

        # Creation of Plot
        self.fig1 = Figure()
        self.addmpl_homodyne(self.fig1)

        # Creation of Plot
        self.fig1 = Figure()
        self.addmpl_vna(self.fig1)

        # To use LATEX in plots
        matplotlib.rc('text', usetex=True)
        matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    
        # IPython console
        self.console = EmbedIPython()
        self.console.kernel.shell.run_cell('%pylab qt')

        self.console.execute("cd ./")

        self.ui.TermVBox.addWidget(self.console)

        self.ui.show()

    def choosePath(self,flag):
        w = QWidget()
        w.resize(320, 240)
        w.setWindowTitle("Select directory where KID files are ")

        if flag == "firm":
            self.firmware = QFileDialog.getOpenFileName(self, "Select Directory")
            self.ui.firmEdit.setText(self.firmware)
            self.gc['FIRMWARE_FILE'] = self.firmware
        elif flag == "vnaPath":
            self.vna_savepath = QFileDialog.getOpenFileName(self, "Select Directory")
            self.ui.vnaEdit.setText(self.vna_savepath)
            self.gc['VNA_SAVEPATH'] = self.vna_savepath
        elif flag == "tarPath":
            self.targ_savepath = QFileDialog.getOpenFileName(self, "Select Directory")
            self.ui.tarEdit.setText(self.targ_savepath)
            self.gc['TARG_SAVEPATH'] = self.targ_savepath
        elif flag == "streamPath":
            self.dirfile_savepath = QFileDialog.getOpenFileName(self, "Select Directory")
            self.ui.streamEdit.setText(self.dirfile_savepath)
            self.gc['DIRFILE_SAVEPATH'] = self.dirfile_savepath

    def chooseFirmPath(self,event):
        self.choosePath("firm")

    def chooseVNAPath(self,event):
        self.choosePath("vnaPath")

    def chooseTargPath(self,event):
        self.choosePath("tarPath")

    def chooseStreamPath(self,event):
        self.choosePath("streamPath")

    def testConn(self,fpga):
        """Tests the link to Roach2 PPC, using return from getFPGA()
            inputs:
                casperfpga object fpga: The fpga object
            outputs: the fpga object"""
        if not fpga:
            try:
                fpga = casperfpga.CasperFpga(self.roach_ip, timeout = 3.)
                # Roach interface
                self.ri = roachInterface(self.fpga, self.gc, self.regs, None)
            except RuntimeError:
                logging.warning("No connection to ROACH. If booting, wait 30 seconds and retry. Otherwise, check gc config.")
        return fpga

    def roach_connection(self,event):
        """Check the connection with ROACH, if it is connected turn green the status icon"""

        self.roach_ip = self.ui.roachIPEdit.toPlainText()

        w = QWidget()

        self.ui.setEnabled(False)
        self.ui.statusbar.showMessage(u'Waiting for roach connection...')
        QMessageBox.information(w, "ROACH Connection", "Starting with ROACH comunication ...")

        try:
            result = self.testConn(self.fpga)
        except:
            result = None
        icon = QIcon()
        if not result:
            icon.addPixmap(QPixmap('./src/icon/wrong_icon.png'))
            self.ui.actionRoach_Status.setIcon(icon)
            self.statusConn = 1
            self.ui.statusbar.showMessage(u'ROACH connection failed!')
            logging.warning('ROACH connection failed.')
            QMessageBox.information(w, "ROACH Connection", "No connection to ROACH. If booting, wait 30 seconds and retry. Otherwise, check gc config.")
        else:
            self.fpga = result
            icon.addPixmap(QPixmap('./src/icon/ok_icon.png'))
            self.ui.actionRoach_Status.setIcon(icon)
            self.statusConn = 0
            self.ui.statusbar.showMessage(u'ROACH connection is successful!')
            logging.info('ROACH connection is successful!')
            QMessageBox.information(w, "ROACH Connection", "Successful communication!")

        self.ui.setEnabled(True)

    def roach_synth(self,event):
        """Synthesizer connection. Check if the synthesizer is connected and set it
            the initial parameters"""

        self.clkFreq = np.float(self.ui.freqClk.toPlainText())
        self.clkPow = np.float(self.ui.powClk.toPlainText())
        self.LOFreq = np.float(self.ui.loFreq.toPlainText())
        self.LOPow = np.float(self.ui.loPow.toPlainText())

        self.synthID = self.ui.comboBox.currentText().upper()

        w = QMessageBox()
        icon = QIcon()

        self.ui.setEnabled(False)
        self.ui.statusbar.showMessage(u'Waiting for synthesizer connection ...')
        QMessageBox.information(w, "Synthesizer Connection", "Starting Synthesizer configuration ...")

        try:
            # Initializing Synthesizer Windfreak
            self.synthRF = synthclass.Synthesizer(self.synthID)
            icon.addPixmap(QPixmap('./src/icon/ok_icon.png'))
            self.ui.actionSynthesizer_status.setIcon(icon)
            self.ui.synthBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: green
                                        }""")
            self.statusSynth = 0
            logging.info('Synthesizer connection is successful')
            self.ui.statusbar.showMessage(u'Synthesizer connection is successful')

            # CLK
            self.synthRF.setControlChannel(0)
            self.synthRF.setPower(True)
            self.synthRF.setRFMute(1)
            self.synthRF.setRFAmp(1)
            self.synthRF.setFrequency(self.clkFreq)
            self.synthRF.setPower(self.clkPow)

            # LO
            self.synthRF.setControlChannel(1)
            self.synthRF.setPower(True)
            self.synthRF.setRFMute(1)
            self.synthRF.setRFAmp(1)
            self.synthRF.setFrequency(self.LOFreq)
            self.synthRF.setPower(self.LOPow)

            QMessageBox.information(w, "Synthesizer connection", "Synthesizer connected and working!")
        except:
            icon.addPixmap(QPixmap('./src/icon/wrong_icon.png'))
            self.ui.actionSynthesizer_status.setIcon(icon)
            self.ui.synthBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: red
                                        }""")
            self.statusSynth = 1
            logging.warning('Synthesizer failed!')
            self.ui.statusbar.showMessage(u'Synthesizer failed!')
            QMessageBox.warning(w, "Synthesizer connection", "Synthesizer connection failed!")

        self.ui.setEnabled(True)

    def qdr_cal(self,event):

        w = QMessageBox()
        icon = QIcon()

        self.ui.setEnabled(False)
        self.ui.statusbar.showMessage(u'Waiting for QDR Calibration ...')
        QMessageBox.information(w, "QDR Calibration", "Starting QDR calibration ...")

        if not self.fpga == None:
            self.fpga.write_int(self.regs['accum_len_reg'], self.ri.accum_len - 1)
            time.sleep(0.1)
            self.fpga.write_int(self.regs['dds_shift_reg'], int(self.gc['dds_shift']))
            time.sleep(0.1)

            # QDR Calibration
            if (self.ri.qdrCal() < 0):
                icon.addPixmap(QPixmap('./src/icon/wrong_icon.png'))
                self.ui.actionQDR_Status.setIcon(icon)
                self.ui.statusbar.showMessage(u'QDR Calibration failed!')
                logging.info('QDR Calibration failed!')
                QMessageBox.information(w, "QDR Calibration", "QDR calibration failed... Check FPGA clock source")
            else:
                icon.addPixmap(QPixmap('./src/icon/ok_icon.png'))
                self.ui.actionQDR_Status.setIcon(icon)
                self.fpga.write_int(self.regs['write_qdr_status_reg'], 1)
                self.ui.statusbar.showMessage(u'QDR Calibration completed!')
                logging.info('QDR Calibration completed!')
                QMessageBox.information(w, "QDR Calibration", "QDR calibration completed!")
        else:
            icon.addPixmap(QPixmap('./src/icon/wrong_icon.png'))
            self.ui.actionQDR_Status.setIcon(icon)
            logging.info('QDR calibration failed... Check ROACH connection')
            QMessageBox.information(w, "QDR Calibration", "QDR calibration failed... Check ROACH connection")

        self.ui.setEnabled(True)

    def roach_atten(self,event):
        """Attenuators connection. Check if the attenuators are connected and calibrate them"""        

        att_ID_1 = int(self.ui.attInIDEdit.toPlainText())
        att_ID_2 = int(self.ui.attOutIDEdit.toPlainText())

        self.attenID = [att_ID_1,  att_ID_2]

        self.att_In = int(self.ui.attInEdit.toPlainText())
        self.att_Out = int(self.ui.attOutEdit.toPlainText())
        self.target_rms = np.float(self.ui.tarLevelEdit.toPlainText())

        w = QMessageBox()
        icon = QIcon()

        self.ui.setEnabled(False)
        self.ui.statusbar.showMessage(u'Waiting for attenuators calibration ... ')
        QMessageBox.information(w, "Attenuation Connection", "Starting input/output attenuators configuration ...")

        try:
            # Attenuation calibration
            att = attCalibration.CalibrationATT(self.attenID)

            if self.fpga:
                att.calibrateADC(self.fpga, self.target_rms, self.att_Out, self.att_In) # ADC level calibration

                icon.addPixmap(QPixmap('./src/icon/ok_icon.png'))
                self.ui.actionRF_Status.setIcon(icon)
                self.ui.attBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: green
                                        }""")
                self.statusAtt = 0
                self.ui.statusbar.showMessage(u'Attenuators connection is succesful')
                logging.info('Attenuators connection is succesful')
                QMessageBox.information(w, "Attenuators connection", "Attenuators connected and working!")
            else:
                icon.addPixmap(QPixmap('./src/icon/wrong_icon.png'))
                self.ui.actionRF_Status.setIcon(icon)
                self.ui.attBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: red
                                        }""")
                self.statusAtt = 0
                self.ui.statusbar.showMessage(u'Attenuators connection failed!')
                logging.warning('Attenuators connection failed!')
                QMessageBox.information(w, "Attenuators connection", "Attenuators calibration failed! Roach is not connected.")

        except:
            icon.addPixmap(QPixmap('./src/icon/wrong_icon.png'))
            self.ui.actionRF_Status.setIcon(icon)
            self.ui.attBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: red
                                        }""")
            self.statusAtt = 1
            self.ui.statusbar.showMessage(u'Attenuators connection failed!')
            logging.warning('Attenuators connection failed! Check attenuators connection.')
            QMessageBox.warning(w, "Attenuators connection", "Attenuators connection failed! Check attenuators connection.")

        self.ui.setEnabled(True)

    def roach_network(self,event):

        self.gc['udp_dest_ip'] = self.ui.ipDstEdit.toPlainText()
        self.gc['udp_dst_port'] = self.ui.portDstEdit.toPlainText()
        self.gc['udp_dest_mac'] = self.ui.macDstEdit.toPlainText()

        self.gc['udp_src_ip'] = self.ui.ipSrcEdit.toPlainText()
        self.gc['udp_src_port'] = self.ui.portSrcEdit.toPlainText()
        self.gc['udp_src_mac'] = self.ui.macSrcEdit.toPlainText()

        self.gc['udp_dest_device'] = self.ui.ethEdit.toPlainText()

        # Update the UDP parameters
        self.eth_port = self.gc['udp_dest_device']
        os.system("sudo ip link set " + self.eth_port + " mtu 9000")

        self.udp_src_ip =  self.gc['udp_src_ip']
        self.udp_src_mac =  self.gc['udp_src_mac']
        self.udp_src_port =  self.gc['udp_src_port']

        self.udp_dst_ip =  self.gc['udp_dest_ip']
        self.udp_dst_mac =  self.gc['udp_dest_mac']
        self.udp_dst_port =  self.gc['udp_dst_port']

        w = QMessageBox()
        icon = QIcon()

        self.ui.setEnabled(False)
        self.ui.statusbar.showMessage(u'UDP configuration ... ')
        QMessageBox.information(w, "UDP Configuration", "Starting UDP configuration ...")

        try:
            # GbE interface
            self.udp = roachDownlink(self.ri, self.fpga, self.gc, self.regs, self.s, self.ri.accum_freq)
            self.udp.configSocket()

            # UDP Configuration
            try:
                self.udp.configDownlink()

                # Register set
                self.fpga.write_int(self.regs['accum_len_reg'], self.ri.accum_len - 1)
                time.sleep(0.1)
                self.fpga.write_int(self.regs['dds_shift_reg'], int(self.gc['dds_shift']))
                time.sleep(0.1)

                icon.addPixmap(QPixmap('./src/icon/ok_icon.png'))
                self.ui.actionNetwork_status.setIcon(icon)
                self.ui.udpConfBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: green
                                        }""")
                self.statusNet = 0
                self.ui.statusbar.showMessage(u'UDP Downlink configured.')
                logging.info('UDP Downlink configured.')
                QMessageBox.information(w, "UDP Downlink", "UDP Network configuraton id done.")

            except AttributeError:
                icon.addPixmap(QPixmap('./src/icon/wrong_icon.png'))
                self.ui.actionNetwork_status.setIcon(icon)
                self.ui.udpConfBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: red
                                        }""")
                self.statusNet = 1
                self.ui.statusbar.showMessage(u'UDP Downlink configuration failed!')
                logging.warning("UDP Downlink could not be configured. Check ROACH connection.")
                QMessageBox.information(w, "UDP Downlink", "UDP Downlink could not be configured. Check ROACH connection.")

        except:
            icon.addPixmap(QPixmap('./src/icon/wrong_icon.png'))
            self.ui.actionNetwork_status.setIcon(icon)
            self.ui.udpConfBtn.setStyleSheet("""QWidget {
                                    color: white;
                                    background-color: red
                                    }""")
            self.statusNet = 1
            self.ui.statusbar.showMessage(u'UDP Network configuraton failed!')
            logging.warning('UDP Network configuraton failed!')
            QMessageBox.information(w, "UDP error", "UDP Network configuraton failed! Check ROACH connection.")

        self.ui.setEnabled(True)

    def write_test_comb(self,event):

        self.min_pos_freq = np.float(self.ui.minPosEdit.toPlainText())*1.0e6
        self.max_pos_freq = np.float(self.ui.maxPosEdit.toPlainText())*1.0e6
        self.min_neg_freq = np.float(self.ui.minNegEdit.toPlainText())*1.0e6
        self.max_neg_freq = np.float(self.ui.maxNegEdit.toPlainText())*1.0e6
        self.symm_offset = np.float(self.ui.offsetEdit.toPlainText())*1.0e6
        self.Nfreq = int(self.ui.nFreqsEdit.toPlainText())

        #w = QMessageBox()

        self.ui.statusbar.showMessage(u'Writting test comb ... ')

        try:
            if self.fpga:
                self.ri.makeFreqComb(self.min_neg_freq,self.max_neg_freq,self.min_pos_freq,self.max_pos_freq,self.symm_offset,self.Nfreq)
                if (len(self.ri.freq_comb) > 400):
                    self.fpga.write_int(self.regs['fft_shift_reg'], 2**5 -1)
                    time.sleep(0.1)
                else:
                    self.fpga.write_int(self.regs['fft_shift_reg'], 2**9 -1)
                    time.sleep(0.1)

                self.ri.upconvert = np.sort(((self.ri.freq_comb + (self.center_freq)*1.0e6))/1.0e6)
                logging.info("RF tones =", self.ri.upconvert)
                self.ri.writeQDR(self.ri.freq_comb, transfunc = False)
                np.save("last_freq_comb.npy", self.ri.freq_comb)

                if not (self.fpga.read_int(self.regs['dds_shift_reg'])):
                    if self.regs['DDC_mixerout_bram_reg'] in self.fpga.listdev():
                        shift = self.ri.return_shift(0)
                        if (shift < 0):
                            self.ui.writeTestBtn.setStyleSheet("""QWidget {
                                                    color: white;
                                                    background-color: red
                                                    }""")
                            self.statusNet = 1
                            self.ui.statusbar.showMessage("Error finding dds shift: Try writing full frequency comb (N = 1000), or single test frequency. Then try again")                    
                            logging.warning("Error finding dds shift: Try writing full frequency comb (N = 1000), or single test frequency. Then try again")
                        else:
                            self.fpga.write_int(self.regs['dds_shift_reg'], shift)

                            self.ui.writeTestBtn.setStyleSheet("""QWidget {
                                                    color: white;
                                                    background-color: green
                                                    }""")
                            self.statusNet = 0
                            self.ui.statusbar.showMessage("Wrote DDS shift (" + str(shift) + ")")
                            logging.info("Wrote DDS shift (" + str(shift) + ")")
                    else:
                        self.fpga.write_int(self.regs['dds_shift_reg'], self.ri.dds_shift)
            else:
                self.ui.writeTestBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: red
                                        }""")
                self.statusNet = 1
                self.ui.statusbar.showMessage(u'Error writting test comb')
                logging.warning('Error writting test comb')
        except KeyboardInterrupt:
            self.ui.writeTestBtn.setStyleSheet("""QWidget {
                                    color: white;
                                    background-color: red
                                    }""")
            self.statusNet = 1
            self.ui.statusbar.showMessage(u'Error writting test comb')
            logging.warning('Error writting test comb')

    def test_udp(self, event):

        w = QMessageBox()

        self.ui.setEnabled(False)
        self.ui.statusbar.showMessage(u'Starting UDP test ... ')
        QMessageBox.information(w, "UDP test", "Starting UDP test ...")

        if self.fpga:
            if (self.udp.testDownlink(5) < 0):
                self.ui.udpTestBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: red
                                        }""")
                self.statusNet = 1
                self.ui.statusbar.showMessage(u'Error receiving data.')
                logging.warning("Error receiving data. Check ethernet configuration.")

            else:
                self.ui.udpTestBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: green
                                        }""")
                self.statusNet = 0
                self.ui.statusbar.showMessage(u'Test successful!')
                logging.warning("Test successful. Connections are working.")

                self.fpga.write_int(self.regs['write_stream_status_reg'], 1)
        else:
                self.ui.udpTestBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: red
                                        }""")
                self.statusNet = 1
                self.ui.statusbar.showMessage(u'Error receiving data.')
                logging.warning("Error receiving data. Check ROACH connection.")            

        self.ui.setEnabled(True)

    def start_plot_VNA(self,event):
        self.plotVNASweep(str(np.load("last_vna_dir.npy")))

    def start_VNA_sweep(self,event):
        self.vnaSweep(self.ri,self.udp,None)

    def vna_sweep_dirfile(self, center_freq = None, save_path = './vna_sweeps', write = None,sweep_dir=None,randomiser=0,samples_per_point=10,num_tones=256,sweep_step=2.5e3,adjust_sideband_leakage=True,auto_fullscale=False,remove_cryostat_input_s21=True,remove_electronics_input_response=True,plot=True,gains=None,step_sleep=0.1):

        write = self.ui.writeTones.isChecked()

        startVNA = np.float(self.ui.startEdit.toPlainText())*1.0e6
        stopVNA = np.float(self.ui.stopEdit.toPlainText())*1.0e6
        center_freq = np.float(self.ui.centralEdit.toPlainText())*1.0e6
        sweep_step = np.float(self.ui.stepEdit.toPlainText())*1.0e6
        num_tones = int(self.ui.nTonesEdit.toPlainText())

        save_path = os.path.join(save_path, sweep_dir)
        bb_freqs, delta_f = np.linspace(startVNA, stopVNA, num_tones, retstep=True)
        
        if randomiser is not None:
            bb_freqs += randomiser

        for ch in range(len(bb_freqs)-1):
            #if np.round(abs(bb_freqs[ch]),-3) in np.around(bb_freqs[ch+1:],-3):


            # AQUI VAMOS *******************************************************************
            if (np.around(abs(bb_freqs[ch])/self.dac_freq_res))*self.dac_freq_res in np.around(bb_freqs[ch+1:]/self.dac_freq_res)*self.dac_freq_res:
                #print '*****FOUND******'
                bb_freqs[ch] += 2*self.dac_freq_res
        
        bb_freqs = np.roll(bb_freqs, - np.argmin(np.abs(bb_freqs)) - 1)
        np.save('./last_bb_freqs.npy',bb_freqs)
        rf_freqs = bb_freqs + center_freq
        np.save('./last_rf_freqs.npy',rf_freqs)
        channels = np.arange(len(rf_freqs))
        np.save('./last_channels.npy',channels)
        #self.v.setFrequencyFast(0,center_freq , 0.01) # LO
        #self.vLO.frequency = center_freq
        self.v.setFrequencyFast( center_freq )

        print '\nVNA baseband freqs (MHz) =', bb_freqs/1.0e6
        print '\nVNA RF freqs (MHz) =', rf_freqs/1.0e6
        if write=='y' or write is True:
            self.writeQDR(bb_freqs,adjust_sideband_leakage=adjust_sideband_leakage,auto_fullscale=auto_fullscale,remove_cryostat_input_s21=remove_cryostat_input_s21,remove_electronics_input_response=remove_electronics_input_response,lo_frequency=center_freq,gains=gains)
        self.fpga.write_int('sync_accum_reset', 0)
        self.fpga.write_int('sync_accum_reset', 1)
        f,i,q = self.sweep_lo_dirfile(Npackets_per = samples_per_point, channels = channels, center_freq = center_freq, span = delta_f, save_path = save_path, bb_freqs=bb_freqs,step = sweep_step, sleep=step_sleep)
        last_vna_dir = save_path
        np.save('./last_vna_dir.npy',np.array([last_vna_dir]))
        np.save('./last_vna_sweep.npy',np.array([f,i,q]))
        #self.plot_kids(save_path = last_vna_dir, bb_freqs = bb_freqs, channels = channels)
        if plot:
            plt.figure('vna-sweep-dirfile')
            for ch in channels:
                plt.plot(f[ch],10*np.log10(i[ch]**2+q[ch]**2))
            plt.show()
        return f,i,q

    def vnaSweep(self, ri, udp, valon):
        """Does a wideband sweep of the RF band, saves data in vna_savepath
           as .npy files
           inputs:
               roachInterface object ri
               gbeConfig object udp
               valon synth object valon
               bool write: Write test comb before sweeping?
               Navg = Number of data points to average at each sweep step"""

        #Navg = np.int(gc[np.where(gc == 'Navg')[0][0]][1])
        Navg = 10

        if not os.path.exists(self.vna_savepath):
            os.makedirs(self.vna_savepath)

        sweep_dir = self.vna_savepath + '/' + \
           str(int(time.time())) + '-' + time.strftime('%b-%d-%Y-%H-%M-%S') + '.dir'

        os.mkdir(sweep_dir)
        np.save("./last_vna_dir.npy", sweep_dir)

        print sweep_dir

        # *** Synthesizer ***
        self.synthRF.setControlChannel(1)
        self.synthRF.setFrequencyFast(self.center_freq)

        span = self.ri.pos_delta

        print "Sweep Span =", 2*np.round(self.ri.pos_delta,2), "Hz"

        start = self.center_freq*1.0e6 - (span)
        stop = self.center_freq*1.0e6 + (span)
        sweep_freqs = np.arange(start, stop, self.lo_step)
        sweep_freqs = np.round(sweep_freqs/self.lo_step)*self.lo_step

        if not np.size(self.ri.freq_comb):
            self.ri.makeFreqComb()
        np.save(sweep_dir + '/bb_freqs.npy', self.ri.freq_comb)
        np.save(sweep_dir + '/sweep_freqs.npy', sweep_freqs)
        Nchan = len(self.ri.freq_comb)

        if not Nchan:
            Nchan = fpga.read_int(self.regs[np.where(self.regs == 'read_comb_len_reg')[0][0]][1])

        for freq in sweep_freqs:
            print 'LO freq =', freq/1.0e6
            self.synthRF.setFrequencyFast(freq)

            self.udp.saveSweepData(Navg, sweep_dir, freq, Nchan,skip_packets = 10)
            time.sleep(0.001)

        self.synthRF.setFrequencyFast(self.center_freq)

        return

    def openStoredSweep(self,savepath):
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

    def plotVNASweep(self,path):

        plt.figure()

        Is, Qs = self.openStoredSweep(path)
        sweep_freqs = np.load(path + '/sweep_freqs.npy')
        bb_freqs = np.load(path + '/bb_freqs.npy')
        rf_freqs = np.zeros((len(bb_freqs),len(sweep_freqs)))

        for chan in range(len(bb_freqs)):
            rf_freqs[chan] = (sweep_freqs + bb_freqs[chan])/1.0e6

        Q = np.reshape(np.transpose(Qs),(len(Qs[0])*(len(sweep_freqs))))
        I = np.reshape(np.transpose(Is),(len(Is[0])*(len(sweep_freqs))))
        mag = np.sqrt(I**2 + Q**2)
        mag = 20*np.log10(mag/np.max(mag))
        mag = np.concatenate((mag[len(mag)/2:],mag[:len(mag)/2]))
        rf_freqs = np.hstack(rf_freqs)
        rf_freqs = np.concatenate((rf_freqs[len(rf_freqs)/2:],rf_freqs[:len(rf_freqs)/2]))

        plt.plot(rf_freqs, mag)
        #plt.plot(mag)

        plt.title(path, size = 16)
        plt.xlabel('frequency (MHz)', size = 16)
        plt.ylabel('dB', size = 16)
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(path,'vna_sweep.png'), dpi = 100, bbox_inches = 'tight')

        plt.show()

        return

    def upload_firmware(self,event):
        w = QWidget()

        self.ui.setEnabled(False)
        QMessageBox.information(w, "ROACH Connection", "Uploading firmware ...")

        try:
            if (self.ri.uploadfpg() < 0):
                self.ui.upFirmBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: red
                                        }""")
                self.statusFirm = 1
                QMessageBox.information(w, "ROACH Firmware", "Firmware upload failed! :(")
            else:
                self.ui.upFirmBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: green
                                        }""")
                self.statusFirm = 0
                QMessageBox.information(w, "ROACH Firmware", "Firmware uploaded successfuly! :)")
        except:
                QMessageBox.information(w, "ROACH Firmware", "Firmware upload failed! :(")
                self.ui.upFirmBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: red
                                        }""")
                self.statusFirm = 1

        self.ui.setEnabled(True)

    def addmpl_homodyne(self,fig):
        self.canvas_H = FigureCanvas(fig)
        self.ui.MainPlot_2.addWidget(self.canvas_H)
        self.canvas_H.draw()
        self.toolbar_H = NavigationToolbar(self.canvas_H,
           self, coordinates=True)
        self.ui.MainPlot_2.addWidget(self.toolbar_H)

    def rmmpl_homodyne(self):
        self.ui.MainPlot_2.removeWidget(self.canvas_H)
        self.canvas_H.close()
        self.ui.MainPlot_2.removeWidget(self.toolbar_H)
        self.toolbar_H.close()

    def addmpl_vna(self,fig):
        self.canvas_V = FigureCanvas(fig)
        self.ui.MainPlot.addWidget(self.canvas_V)
        self.canvas_V.draw()
        self.toolbar_V = NavigationToolbar(self.canvas_V,
           self, coordinates=True)
        self.ui.MainPlot.addWidget(self.toolbar_V)

    def rmmpl_vna(self):
        self.ui.MainPlot.removeWidget(self.canvas_V)
        self.canvas_V.close()
        self.ui.MainPlot.removeWidget(self.toolbar_V)
        self.toolbar_V.close()

app = QtGui.QApplication(sys.argv)
MyWindow = MainWindow()
sys.exit(app.exec_())
