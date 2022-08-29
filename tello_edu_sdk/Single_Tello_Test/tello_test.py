from tello import Tello
import sys
from datetime import datetime
import time
import socket as sock

start_time = str(datetime.now())

file_name = sys.argv[1]

f = open(file_name, "r")
commands = f.readlines()

tello = Tello()

'''
local_ip = ''
local_port = 8890
socket = sock.socket(sock.AF_INET, sock.SOCK_DGRAM)  # socket for sending cmd
socket.bind((local_ip, local_port))

tello_ip = '192.168.10.1'
tello_port = 8889
tello_adderss = (tello_ip, tello_port)

#socket.sendto('command'.encode('utf-8'), tello_adderss)
'''
for command in commands:
    if command != '' and command != '\n':
        command = command.rstrip()
        if command.find('delay') != -1:
            sec = float(command.partition('delay')[2])
            print 'delay %s' % sec
            time.sleep(sec)
            pass
        else:
            tello.send_command(command)
            #response, ip = socket.recvfrom(1024)
            #print(response)


log = tello.get_log()

out = open('log/' + '1' + '.txt', 'w')
for stat in log:
    stat.print_stats()
    str = stat.return_stats()
    out.write(str)
