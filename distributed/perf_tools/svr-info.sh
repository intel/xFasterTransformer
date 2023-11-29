#!/bin/bash
set -x

tar -zxvf svr-info.tgz

mkdir -p svr_log

sudo ./svr-info/svr-info -output ./svr_log