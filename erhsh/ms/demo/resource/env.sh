# LOG Level
export SLOG_PRINT_TO_STDOUT=1
export GLOG_logtostderr=1
export GLOG_v=1

#
LOCAL_USER=~
LOCAL_ASCEND=/usr/local/Ascend

#
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/driver/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/fwkacllib/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/add-ons:${LD_LIBRARY_PATH}

#
export TBE_IMPL_PATH=${LOCAL_ASCEND}/opp/op_impl/built-in/ai_core/tbe
export PATH=${LOCAL_ASCEND}/fwkacllib/ccec_compiler/bin/:${PATH}
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}

#
export MSLIBS_SERVER=10.10.10.10
export MSLIBS_CACHE_PATH=${LOCAL_USER}/.mslib
export MID=MDAzNjEK
export MWS=${LOCAL_USER}/workspace

#
export RANK_TABLE_FILE=${MWS}/rank_table_8p.json
export MINDSPORE_HCCL_CONFIG_PATH=${RANK_TABLE_FILE}

#
export CUDA_HOME=/usr/local/cuda-10.1
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

#
alias sc=source
alias bs=base64
alias tailf="tail -f"
alias gs="git status"
