#-*- encoding:utf-8 -*-
# Author: wenmeng.zwm@alibaba-inc.com
# Date: 2019-10-12
# util to hanlde python2 python3 compatibility

import sys

def in_python2():
  return sys.version_info[0] == 2

def in_python3():
  return sys.version_info[0] == 3
