# -*- coding: utf-8 -*-
"""
Created on Wed Aug 08 08:10:27 2018

@author: spxtg
"""
import sys
import numpy as np
import socket as sock
import time
import struct
import select
##


""" === FPGA object === """

def getFPGA(ip):
    
    """ Returns a casperfpga object of the Roach2 """

    try:
        fpga = casperfpga.katcp_fpga.KatcpFpga(ip, timeout = 120.)
    except RuntimeError:
        print "\nNo connection to ROACH. If booting, wait 30 seconds and retry."
    return fpga


""" === KATCP connection through FPGA object === """
class connectManager(object):

    def __init__(self, fpga, socket, data_rate, udpParam):
        self.fpga = fpga
        self.sk = socket
        self.data_rate = data_rate
        self.data_len = 8192
        self.header_len = 42
        self.buf_size = self.data_len + self.header_len
        self.temp_dstip = udpParam[0]
        self.udp_dest_ip = self.temp_dstip
        self.temp_srcip = udpParam[1]
        self.udp_src_ip = self.temp_srcip
        self.udp_src_port = udpParam[2]
        self.udp_dst_port = udpParam[3]
        src_mac = udpParam[4]
        self.udp_srcmac1 = int(src_mac[0:4], 16)
        self.udp_srcmac0 = int(src_mac[4:], 16)
        dest_mac = udpParam[5]
        self.udp_destmac1 = int(dest_mac[0:4], 16)
        self.udp_destmac0 = int(dest_mac[4:], 16)
#
    
    def configSocket(self):
        """Configure socket parameters"""
        dest_ip = self.udp_dest_ip
        dest_port = self.udp_dst_port
        
#        size = self.sk.getsockopt(sock.SOL_SOCKET, sock.SO_RCVBUF)
#        print
#        while size>0:
#            r = self.sk.recv(65536)
#            l=len(r)
#            if l==0:
#                break
#            size -= l
#            print '\rclearing %d'%size,
#        print
        
        try:
            #self.sk.setsockopt(sock.SOL_SOCKET, sock.SO_RCVBUF, self.buf_size)
            #get max buffer size, buffers are good for not dropping packets
            
            with open('/proc/sys/net/core/rmem_max', 'r') as f:
                buf_max = int(f.readline())
            
#            self.sk.setsockopt(sock.SOL_SOCKET, sock.SO_REUSEADDR, 1)
#            self.sk.bind((udp_src_ip, udp_src_port)) 
            print buf_max
            self.sk.setsockopt(sock.SOL_SOCKET, sock.SO_RCVBUF, buf_max)
            self.sk.setblocking(0)
#            self.sk.bind(('eth0', 3))
            self.sk.bind((dest_ip, dest_port))
#            self.sk.setsockopt(sock.SOL_SOCKET, sock.SO_REUSEADDR, 1)
#            self.sk.bind((str(dest_ip), dest_port))
#             ??? self.sk.bind('udp_dest_device', 0) # for eth - 0 -
        except sock.error, v:
            errorcode = v[0]
            if errorcode == 19:
                print "Ethernet device could not be found"
                pass
            
        return
#

    def configDownlink(self):
        """Configure GbE parameters"""
        
        self.fpga.write_int('GbE_tx_srcmac0', self.udp_srcmac0) # 'udp_srcmac0_reg', self.udp_srcmac0
        time.sleep(0.05)
        self.fpga.write_int('GbE_tx_srcmac1', self.udp_srcmac1) # 'udp_srcmac1_reg', self.udp_srcmac1
        time.sleep(0.05)
        self.fpga.write_int('GbE_tx_destmac0', self.udp_destmac0) # 'udp_destmac0_reg', self.udp_destmac0
        time.sleep(0.05)
        self.fpga.write_int('GbE_tx_destmac1', self.udp_destmac1) # 'udp_destmac1_reg', self.udp_destmac1
        time.sleep(0.05)
        
        temp_srcip = sock.inet_aton(self.temp_srcip)
        self.srcip = struct.unpack(">L", temp_srcip)[0]
#        self.srcip = 192*(2**24) + 168*(2**16) + 40*(2**8) + 96
        self.fpga.write_int('GbE_tx_srcip', self.srcip) # 'udp_srcip_reg', self.udp_src_ip
        time.sleep(0.05)
        
        temp_dstip = sock.inet_aton(self.temp_dstip)
        self.dstip = struct.unpack(">L", temp_dstip)[0]
#        self.dstip = 192*(2**24) + 168*(2**16) + 40*(2**8) + 1
        self.fpga.write_int('GbE_tx_destip', self.dstip) # 'udp_destip_reg', self.udp_dest_ip
        time.sleep(0.1)
        
        self.fpga.write_int('GbE_tx_destport', self.udp_dst_port) # 'udp_destport_reg', self.udp_dst_port
        time.sleep(0.1)
        self.fpga.write_int('GbE_tx_srcport', self.udp_src_port) # 'udp_srcport_reg', self.udp_src_port
        time.sleep(0.1)
        self.fpga.write_int('GbE_pps_start', 0) # 'udp_start_reg', 0
        time.sleep(0.1)
        self.fpga.write_int('GbE_pps_start', 1) # 'udp_start_reg', 1
        time.sleep(0.1)
        self.fpga.write_int('GbE_pps_start', 0) # 'udp_start_reg', 0
        time.sleep(0.1)
        self.fpga.write_int('PFB_fft_shift', 2**5-1)
        time.sleep(0.1)
        self.fpga.write_int('GbE_rx_ack', 1)
        time.sleep(0.1)
        self.fpga.write_int('GbE_rx_rst', 0)
        time.sleep(0.1)
        self.fpga.write_int('GbE_tx_rst', 0)
        time.sleep(0.1)
        self.fpga.write_int('GbE_tx_rst', 1)
        time.sleep(0.1)
        self.fpga.write_int('GbE_tx_rst', 0)
        
        print "Downlink configured."
        
        return
#
        
    def waitForData(self):
        """Uses select function to poll data socket
           outputs:
               packet: UDP packet, string packed"""
        timeout = 10
        read, write, error = select.select([self.sk], [], [], timeout)
        if not (read or write or error):
            print "Socket timed out"
            return
        else:
            print "waitfordata running"
            for sk in read:
                packet = self.sk.recv(self.buf_size)
                if len(packet) != self.buf_size:
                    packet = []
        return packet
#
        
    def zeroPPS(self):
        """Sets the PPS counter to zero"""
        self.fpga.write_int('GbE_pps_start', 0)
        time.sleep(0.1)
        self.fpga.write_int('GbE_pps_start', 1)
        time.sleep(0.1)
        return
#
        
    def parsePacketData(self):
        """Parses packet data, filters reception based on source IP
           outputs:
               packet: The original data packet
               float data: Array of channel data
               header: String packed IP/ETH header
               saddr: The packet source address"""
        packet = self.waitForData()
        if not packet:
            print "Non-Roach packet received"
            return
        data = np.fromstring(packet[self.header_len:], dtype = '<i').astype('float')
        print data
        print "OK"
        header = packet[:self.header_len]
        print header
        print "OK"
        saddr = np.fromstring(header[26:30], dtype = "<I")
        saddr = sock.inet_ntoa(saddr) # source addr
        print saddr
        print "OK"
        ### Filter on source IP ###
        if (saddr != self.udp_src_ip):
            print "Non-Roach packet received"
            return
        return packet, data, header, saddr
#
    
    def testDownlink(self, time_interval):
        """Tests UDP link. Monitors data stream for time_interval and checks
           for packet count increment
           inputs:
               float time_interval: time interval to monitor, seconds
           outputs:
               returns 0 on success, -1 on failure"""
        print "Testing downlink..."
        first_idx = np.zeros(1)
        self.zeroPPS()
        Npackets = np.ceil(time_interval * self.data_rate)
        print "Npackets = ", Npackets
        count = 0
        while count < Npackets:
            try:
                packet, data, header, saddr = self.parsePacketData()
            except TypeError:
                continue
            print "Done!"
            if not packet:
                print "No packet"
                continue
            else:
                packet_count = (np.fromstring(packet[-4:],dtype = '>I'))
                print packet_count
            count += 1
            print "Count is ", count
        if (packet_count - first_idx < 1):
            return -1
        return 0