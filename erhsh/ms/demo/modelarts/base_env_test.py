import os
import sys
import platform


print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Python infos: <<<<<<<<<<<<<<<<<<<<<<<<<")

print("-------------------------- sys.path:")
for p in sys.path:
    print(p)

print("-------------------------- sys.executable:")
print(sys.executable)

print("-------------------------- __file__")
print(__file__)

print("-------------------------- os.getpwd():")
print(os.getcwd())

print("-------------------------- platform.uname():")
uname = platform.uname()
print(uname)

print("-------------------------- list cur dir:")
if "Win" in uname.system:
    os.system("dir .")
else:
    os.system("ls -al")

print('-------------------------- pip list:')
os.system("pip list -v")

print('-------------------------- mindspore ver:')
os.system("cat `pip show pip | grep Location | awk '{print $2}'`/mindspore/.commit_id")

print("-------------------------- ping baidu:")
if "Win" in uname.system:
    os.system("ping www.baidu.com")
else:
    os.system("ping -c 4 www.baidu.com")

print("-------------------------- ENV:")
print(os.environ)