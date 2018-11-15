#!/usr/bin/python

import sys
import config_reader
import os.path as path
import wget
import pathlib
import json


def read_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def download(argv):
   config = read_config()
   if len(argv) == 0:
       argv = config['devices'].keys()
   for device in argv:
       download_device(device, config)
       download_gafgyt(device, config)
       download_mirai(device, config)

def download_device(device, config):
    if config['devices'][device] is None:
        print(f"Device {device} does not exist in the config!")
        return
    filename = f"{device}/benign_traffic.csv"
    if path.exists('data/' + filename):
        print(f"{device} data is already downloaded!")
        return
    pathlib.Path('data/' + device).mkdir(parents=True, exist_ok=True) 
    url = config['data_url'] + filename
    print(url)
    wget.download(url, 'data/' + filename)

def download_gafgyt(device, config):
    filename = f"{device}/gafgyt_attacks.rar"
    if path.exists('data/' + filename):
        print(f"{device} gafgyt attack data is already downloaded!")
        return
    pathlib.Path('data/' + device).mkdir(parents=True, exist_ok=True) 
    url = config['data_url'] + filename
    print(url)
    wget.download(url, 'data/' + filename)
    #you need to unpack rar files yourself, it is problematic in python

def download_mirai(device, config):
    if device == 'Ennio_Doorbell' or device == 'Samsung_SNH_1011_N_Webcam':
        #does not have mirai
        return
    filename = f"{device}/mirai_attacks.rar"
    if path.exists('data/' + filename):
        print(f"{device} mirai attack data is already downloaded!")
        return
    pathlib.Path('data/' + device).mkdir(parents=True, exist_ok=True) 
    url = config['data_url'] + filename
    print(url)
    wget.download(url, 'data/' + filename)
    #you need to unpack rar files yourself, it is problematic in python

if __name__ == "__main__":
   download(sys.argv[1:])
